# Combo variant of a395ee82 (index 5)
# combo_spec: D6/0+D7/1+D11/0
# Name: 面积间距八连通重心

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, deque

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# combo_rule:
#   - 输入仍然只有两类对象: 一个较大的单色模板 T, 以及 5-6 个单例像素构成的晶格标记
#     (唯一一个 T 色点作为原点, 其余点为 O 色)。
#   - 融合后的关键变化不再是彼此独立的:
#       1. 模板面积 A 决定统一步长 s = ceil(sqrt(A))。
#       2. 单例像素之间的间隔也用这个 s, 所以 solver 必须先看模板面积, 才能把像素偏移还原成粗晶格偏移。
#       3. 粗晶格要求是 8 连通, 因而允许纯对角相邻, 输出里会出现斜向 stamp。
#       4. 粗晶格锚点仍从输入模板 bbox 左上角平移得到, 但这个锚点现在解释为“模板重心应该落到的位置”。
#          因此真正的 stamp 左上角 = round((r0 + dr*s, c0 + dc*s) - centroid(template)).
#   - 这让面积、拓扑、重心三者共同影响输出: 面积改变 s, 8 连通改变哪些 (dr,dc) 出现,
#     重心决定同一批锚点如何回落到整数像素。
#
# concrete_trace:
#   - 例: 模板 bbox 为 4x3, 面积 A=10, 则 s=ceil(sqrt(10))=4。设模板局部重心约为 (1.6, 1.0)。
#   - 输入单例若出现在 {(0,0), (4,4), (8,4)} 这类以 4 为间隔的位置, solver 先用 A 推出 s=4,
#     再把它们除以 4 得到粗偏移 {(0,0), (1,1), (2,1)}。
#   - 原题会把 (1,1) 副本放到 bbox 左上角平移 (h,w) 的位置; 现在先得到锚点 (r0+4, c0+4),
#     再把这个锚点当作“重心目标”, 所以副本左上角是 round(r0+4-1.6, c0+4-1.0)。
#   - 与原题相比, 单例簇的间隔、输出副本的平移尺度、以及副本相对输入模板的整体偏移都会变化。
#
# complexity_calibration:
#   - 在原 spec 的面积范围 6-10 内, s 只有 3 和 4 两档, 复杂度有限且能在样例里重复出现。
#   - 8 连通若只“允许”而不强制, 很容易退化成原来的 4 连通分布, 所以 generator 强制使用
#     8 连通且至少含对角接触、并且不是 4 连通的 lattice。
#   - 为了让 D11 不被抵消, generator 还会拒绝那些组合规则输出与 bbox 锚点版本几乎相同的样本。
#
# implicit_assumptions_review:
#   - template = max(comps, key=area):
#       仍要求模板是最大组件; generator 保持“一个模板 + 若干单例”结构。
#   - find_components(..., connectivity=4):
#       仍然安全, 因为单例间隔至少 3, 即使粗晶格 8 连通, 像素层面也不会粘连成大组件。
#   - 原代码通过 gcd 反推 gunit:
#       在本融合里不再成立为主规则, 因为步长必须来自模板面积; 因此 main() 改成先算面积, 再检查
#       所有单例偏移都能被 s 整除。
#   - 原代码的 off_r = dr*h, off_c = dc*w:
#       这里被替换成“先走面积步长 s, 再按重心回落到左上角”的几何解释。
#   - 重心锚点引入了新的假设:
#       所有按该规则生成的 stamp 必须完整落在网格内; generator 会先筛 r0/c0 的可行范围。
#
# placement_legality:
#   - generator 先枚举合法模板和 8 连通 lattice, 再用新规则筛可行的输入模板位置。
#   - 它保证:
#       1. 输入模板本体在 22x22 内;
#       2. 按组合规则生成的所有输出 stamp 在 22x22 内;
#       3. 输出 stamp 之间的重叠很少, 保持画面整洁;
#       4. 单例簇 bbox 与输入模板 bbox 至少隔 1 个背景像素;
#       5. 单例不会落在任何未来 stamp 像素上;
#       6. main(input) 与预期输出完全一致。
#
# observability_and_cleanliness:
#   - 人类可以先看输入模板的面积, 观察单例簇是以 3 或 4 的步长排布, 再在输出里看到相同步长主导
#     的稠密/稀疏差异。
#   - 因为 lattice 强制包含对角关系, 输出会稳定出现斜向副本, 不会退化成单纯十字展开。
#   - 颜色规则仍然简单: T 是普通 stamp 色, O 只负责原点副本和重叠区域; 不依赖颜色数值。
#   - 生成器限制输出重叠数量, 避免大面积糊成一团, 让“面积步长 + 斜向排列 + 重心偏移”能被肉眼区分。
#
# new_main_assumptions_checked_against_generator:
#   - 非模板组件全部是面积 1 的单例。
#   - 单例数量仍为 5-6。
#   - 唯一 T 色单例是原点, 其余单例主色为 O。
#   - 以面积推出的步长 s 能整除所有单例偏移。
#   - 归一化后的粗晶格包含 (0,0), 且 8 连通, 原点位于其 bbox 严格内部。
#   - generator 通过候选枚举和最终回放校验来保证这些前提。


GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
TARGET_STEPS = [3, 4]

PLACEMENT_RETRIES = 120
SINGLETON_RETRIES = 120
GENERATE_RETRIES = 500
MAX_OVERLAP_PIXELS = 2

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]


def _round_nearest(x: float) -> int:
    return int(math.floor(float(x) + 0.5))


def _area_step(area: int) -> int:
    return int(math.ceil(math.sqrt(int(area))))


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


def _has_diagonal_contact(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> bool:
    pts = {tuple(map(int, cell)) for cell in cells}
    for r, c in pts:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            if (int(r + dr), int(c + dc)) in pts:
                return True
    return False


def _template_centroid(templ_rel: list[tuple[int, int]]) -> tuple[float, float]:
    rs = [int(r) for r, _c in templ_rel]
    cs = [int(c) for _r, c in templ_rel]
    return float(sum(rs) / len(rs)), float(sum(cs) / len(cs))


def _combo_shift(
    r0: int,
    c0: int,
    dr: int,
    dc: int,
    step: int,
    centroid_r: float,
    centroid_c: float,
) -> tuple[int, int]:
    anchor_r = int(r0 + dr * step)
    anchor_c = int(c0 + dc * step)
    return _round_nearest(anchor_r - centroid_r), _round_nearest(anchor_c - centroid_c)


def _area_bbox_shift(
    r0: int,
    c0: int,
    dr: int,
    dc: int,
    step: int,
    _centroid_r: float,
    _centroid_c: float,
) -> tuple[int, int]:
    return int(r0 + dr * step), int(c0 + dc * step)


def _original_bbox_shift(
    r0: int,
    c0: int,
    dr: int,
    dc: int,
    h: int,
    w: int,
    _centroid_r: float,
    _centroid_c: float,
) -> tuple[int, int]:
    return int(r0 + dr * h), int(c0 + dc * w)


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(int(dr) for dr, _dc in deltas)
    max_dr = max(int(dr) for dr, _dc in deltas)
    min_dc = min(int(dc) for _dr, dc in deltas)
    max_dc = max(int(dc) for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _render_output(
    bg: int,
    H: int,
    W: int,
    templ_rel: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
    templ_col: int,
    overlap_col: int,
    shift_fn,
) -> tuple[np.ndarray, int]:
    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc in deltas:
        shift_r, shift_c = shift_fn(int(dr), int(dc))
        is_origin = int(dr) == 0 and int(dc) == 0
        for rr, cc in templ_rel:
            R = int(shift_r + rr)
            C = int(shift_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    center[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out, int(np.count_nonzero(total >= 2))


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_coords = template["coords"]
    templ_area = int(template["area"])
    templ_col = int(g[int(templ_coords[0, 0]), int(templ_coords[0, 1])])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))
    templ_rel = [(int(r) - int(r0), int(c) - int(c0)) for r, c in templ_coords]
    centroid_r, centroid_c = _template_centroid(templ_rel)
    step = int(_area_step(templ_area))

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        assert int(comp["area"]) == 1
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    assert SINGLETON_COUNT_RANGE[0] <= len(singletons) <= SINGLETON_COUNT_RANGE[1]

    origin = [(int(r), int(c)) for r, c, col in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for _r, _c, col in singletons if int(col) != int(templ_col)]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas_with_col: list[tuple[int, int, int]] = []
    for r, c, col in singletons:
        dr = int(r - orr)
        dc = int(c - orc)
        assert dr % step == 0 and dc % step == 0
        deltas_with_col.append((int(dr // step), int(dc // step), int(col)))

    coarse_cells = [(int(dr), int(dc)) for dr, dc, _col in deltas_with_col]
    assert (0, 0) in coarse_cells
    assert _is_connected(coarse_cells, connectivity=8)
    rs = [int(dr) for dr, _dc in coarse_cells]
    cs = [int(dc) for _dr, dc in coarse_cells]
    assert min(rs) < 0 < max(rs)
    assert min(cs) < 0 < max(cs)

    return {
        "bg": bg,
        "H": H,
        "W": W,
        "templ_rel": templ_rel,
        "templ_area": templ_area,
        "templ_col": templ_col,
        "overlap_col": overlap_col,
        "r0": r0,
        "c0": c0,
        "h": h,
        "w": w,
        "step": step,
        "centroid_r": centroid_r,
        "centroid_c": centroid_c,
        "deltas": [(int(dr), int(dc)) for dr, dc, _col in deltas_with_col],
    }


def _solve_with_mode(grid: np.ndarray, *, mode: str) -> np.ndarray:
    scene = _extract_scene(grid)
    bg = int(scene["bg"])
    H = int(scene["H"])
    W = int(scene["W"])
    templ_rel = list(scene["templ_rel"])
    templ_col = int(scene["templ_col"])
    overlap_col = int(scene["overlap_col"])
    r0 = int(scene["r0"])
    c0 = int(scene["c0"])
    h = int(scene["h"])
    w = int(scene["w"])
    step = int(scene["step"])
    centroid_r = float(scene["centroid_r"])
    centroid_c = float(scene["centroid_c"])
    deltas = list(scene["deltas"])

    if mode == "combo":
        shift_fn = lambda dr, dc: _combo_shift(r0, c0, dr, dc, step, centroid_r, centroid_c)
    elif mode == "area_bbox":
        shift_fn = lambda dr, dc: _area_bbox_shift(r0, c0, dr, dc, step, centroid_r, centroid_c)
    elif mode == "original":
        shift_fn = lambda dr, dc: _original_bbox_shift(r0, c0, dr, dc, h, w, centroid_r, centroid_c)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out, _ = _render_output(bg, H, W, templ_rel, deltas, templ_col, overlap_col, shift_fn)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp copies of a monochrome template from an area-scaled 8-neighbor singleton lattice,
    aligning each copy by the template centroid.

    Input structure:
    - The background is uniform and one largest monochrome component in color T is the template.
    - Elsewhere sit 5-6 isolated singleton pixels: exactly one is also color T and marks the
      lattice origin, while the others are color O.
    - The singleton spacing is not fixed. It equals a function of template area, and after
      dividing by that spacing the recovered coarse lattice is 8-connected, so diagonal branches
      are allowed.

    Rule:
    - Count the template pixels to obtain area A, then compute the shared lattice step
      s = ceil(sqrt(A)).
    - Measure every singleton offset from the unique T-colored singleton and divide by s to
      recover coarse offsets (dr, dc).
    - For each coarse offset, move the input template's bbox anchor by (dr*s, dc*s), treat that
      integer location as the target position of the template centroid, and round back to the
      stamp's top-left corner.
    - Paint every stamp in T, except the origin copy which becomes O, and recolor any overlap
      region to O as well.

    Key insight:
    - The solver has to fuse three visible cues: template area reveals the step size, the
      singleton layout reveals an 8-connected coarse shape, and the template's own centroid tells
      how those coarse anchors convert into actual stamp pixels.
    """
    return _solve_with_mode(grid, mode="combo")


def _enumerate_template_candidates() -> dict[int, list[dict[str, object]]]:
    out: dict[int, list[dict[str, object]]] = {step: [] for step in TARGET_STEPS}
    for h in TEMPLATE_H_CHOICES:
        total = h * TEMPLATE_W
        lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
        hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
        coords = [(int(r), int(c)) for r in range(h) for c in range(TEMPLATE_W)]
        for area in range(lo, hi + 1):
            step = int(_area_step(area))
            if step not in out:
                continue
            for cells in itertools.combinations(coords, area):
                if not _touches_all_edges(cells, h, TEMPLATE_W):
                    continue
                if not _is_connected(cells, connectivity=4):
                    continue
                templ_cells = [(int(r), int(c)) for r, c in cells]
                centroid_r, centroid_c = _template_centroid(templ_cells)
                out[step].append(
                    {
                        "h": h,
                        "w": TEMPLATE_W,
                        "cells": templ_cells,
                        "area": area,
                        "centroid_r": centroid_r,
                        "centroid_c": centroid_c,
                    }
                )
    return out


def _enumerate_lattice_candidates() -> dict[int, list[dict[str, object]]]:
    out: dict[int, list[dict[str, object]]] = {
        n: [] for n in range(SINGLETON_COUNT_RANGE[0], SINGLETON_COUNT_RANGE[1] + 1)
    }
    for rows in range(LATTICE_EXTENT_RANGE[0], LATTICE_EXTENT_RANGE[1] + 1):
        for cols in range(LATTICE_EXTENT_RANGE[0], LATTICE_EXTENT_RANGE[1] + 1):
            coords = [(int(r), int(c)) for r in range(rows) for c in range(cols)]
            for n in range(SINGLETON_COUNT_RANGE[0], SINGLETON_COUNT_RANGE[1] + 1):
                for cells in itertools.combinations(coords, n):
                    if not _touches_all_edges(cells, rows, cols):
                        continue
                    if not _is_connected(cells, connectivity=8):
                        continue
                    if _is_connected(cells, connectivity=4):
                        continue
                    if not _has_diagonal_contact(cells):
                        continue
                    interiors = _interior_cells(cells, rows, cols)
                    if not interiors:
                        continue
                    out[n].append(
                        {
                            "cells": [(int(r), int(c)) for r, c in cells],
                            "interiors": interiors,
                            "rows": rows,
                            "cols": cols,
                        }
                    )
    return out


TEMPLATE_CANDIDATES = _enumerate_template_candidates()
LATTICE_CANDIDATES = _enumerate_lattice_candidates()
assert all(TEMPLATE_CANDIDATES[step] for step in TARGET_STEPS)
assert all(LATTICE_CANDIDATES[n] for n in range(SINGLETON_COUNT_RANGE[0], SINGLETON_COUNT_RANGE[1] + 1))


def _valid_template_origins(
    deltas: list[tuple[int, int]],
    h: int,
    w: int,
    step: int,
    centroid_r: float,
    centroid_c: float,
) -> tuple[list[int], list[int]]:
    valid_r0: list[int] = []
    valid_c0: list[int] = []

    for r0 in range(GRID_SIZE - h + 1):
        ok = True
        for dr, _dc in deltas:
            shift_r, _ = _combo_shift(r0, 0, dr, 0, step, centroid_r, centroid_c)
            if shift_r < 0 or shift_r + h - 1 >= GRID_SIZE:
                ok = False
                break
        if ok:
            valid_r0.append(int(r0))

    for c0 in range(GRID_SIZE - w + 1):
        ok = True
        for _dr, dc in deltas:
            _, shift_c = _combo_shift(0, c0, 0, dc, step, centroid_r, centroid_c)
            if shift_c < 0 or shift_c + w - 1 >= GRID_SIZE:
                ok = False
                break
        if ok:
            valid_c0.append(int(c0))

    return valid_r0, valid_c0


def _stamp_occupancy(
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    deltas: list[tuple[int, int]],
    step: int,
    centroid_r: float,
    centroid_c: float,
) -> tuple[set[tuple[int, int]], int]:
    counts: Counter[tuple[int, int]] = Counter()
    for dr, dc in deltas:
        shift_r, shift_c = _combo_shift(r0, c0, dr, dc, step, centroid_r, centroid_c)
        for tr, tc in templ_cells:
            counts[(int(shift_r + tr), int(shift_c + tc))] += 1
    overlap = sum(1 for v in counts.values() if int(v) >= 2)
    return set(counts.keys()), int(overlap)


def _place_singletons(
    deltas: list[tuple[int, int]],
    step: int,
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
    future_pixels: set[tuple[int, int]],
    expected: np.ndarray,
) -> np.ndarray | None:
    orr_lo, orr_hi = -min_dr * step, GRID_SIZE - 1 - max_dr * step
    orc_lo, orc_hi = -min_dc * step, GRID_SIZE - 1 - max_dc * step
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(r0), int(r0 + h - 1)
    tc0, tc1 = int(c0), int(c0 + w - 1)

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(int(orr_lo), int(orr_hi))
        orc = random.randint(int(orc_lo), int(orc_hi))

        sr0 = int(orr + min_dr * step)
        sr1 = int(orr + max_dr * step)
        sc0 = int(orc + min_dc * step)
        sc1 = int(orc + max_dc * step)
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        positions = [(int(orr + dr * step), int(orc + dc * step)) for dr, dc in deltas]
        if any(pos in future_pixels for pos in positions):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
        for tr, tc in templ_cells:
            grid[int(r0 + tr), int(c0 + tc)] = int(templ_col)
        for dr, dc in deltas:
            rr = int(orr + dr * step)
            cc = int(orc + dc * step)
            grid[rr, cc] = int(templ_col) if (int(dr) == 0 and int(dc) == 0) else int(overlap_col)

        try:
            produced = main(grid)
        except Exception:
            continue
        if np.array_equal(produced, expected):
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    bg, templ_col, overlap_col = random.sample(range(10), 3)
    target_step = random.choice(TARGET_STEPS)

    templ_spec = random.choice(TEMPLATE_CANDIDATES[target_step])
    templ_cells = list(templ_spec["cells"])
    h = int(templ_spec["h"])
    w = int(templ_spec["w"])
    area = int(templ_spec["area"])
    step = int(_area_step(area))
    centroid_r = float(templ_spec["centroid_r"])
    centroid_c = float(templ_spec["centroid_c"])
    if step != target_step:
        return None

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lattice_spec = random.choice(LATTICE_CANDIDATES[n_sing])
    lattice_cells = list(lattice_spec["cells"])
    origin = random.choice(list(lattice_spec["interiors"]))
    deltas = [(int(r - origin[0]), int(c - origin[1])) for r, c in lattice_cells]
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    valid_r0, valid_c0 = _valid_template_origins(deltas, h, w, step, centroid_r, centroid_c)
    if not valid_r0 or not valid_c0:
        return None

    expected = None
    future_pixels: set[tuple[int, int]] | None = None

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.choice(valid_r0)
        c0 = random.choice(valid_c0)

        future_pixels, overlap_pixels = _stamp_occupancy(
            templ_cells, r0, c0, deltas, step, centroid_r, centroid_c
        )
        if overlap_pixels > MAX_OVERLAP_PIXELS:
            continue

        combo_out, combo_overlap = _render_output(
            bg,
            GRID_SIZE,
            GRID_SIZE,
            templ_cells,
            deltas,
            templ_col,
            overlap_col,
            lambda dr, dc: _combo_shift(r0, c0, dr, dc, step, centroid_r, centroid_c),
        )
        if combo_overlap > MAX_OVERLAP_PIXELS:
            continue

        area_bbox_out, _ = _render_output(
            bg,
            GRID_SIZE,
            GRID_SIZE,
            templ_cells,
            deltas,
            templ_col,
            overlap_col,
            lambda dr, dc: _area_bbox_shift(r0, c0, dr, dc, step, centroid_r, centroid_c),
        )
        original_out, _ = _render_output(
            bg,
            GRID_SIZE,
            GRID_SIZE,
            templ_cells,
            deltas,
            templ_col,
            overlap_col,
            lambda dr, dc: _original_bbox_shift(r0, c0, dr, dc, h, w, centroid_r, centroid_c),
        )

        diff_vs_area_bbox = int(np.count_nonzero(combo_out != area_bbox_out))
        diff_vs_original = int(np.count_nonzero(combo_out != original_out))
        if diff_vs_area_bbox < max(4, len(templ_cells) // 2):
            continue
        if diff_vs_original < max(8, len(templ_cells)):
            continue

        expected = combo_out
        break

    if expected is None or future_pixels is None:
        return None

    return _place_singletons(
        deltas=deltas,
        step=step,
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
        future_pixels=future_pixels,
        expected=expected,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
