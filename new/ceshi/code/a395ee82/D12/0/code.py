# Variant of a395ee82
# Dimension: D12 — 多步处理序列 (multi_step)
# Change: 单步处理 — 直接从输入计算输出(识别模板、解析晶格、stamp所有副本)... → 两模板交替stamp

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis
# a) 具体化与行为追踪:
#    - 输入不再只有一个大模板，而是有两个分离的单色模板 A/B；二者 bbox 尺寸相同，但形状不同、颜色不同。
#    - 仍有一个单点晶格：其中恰有一个单点颜色属于 {A,B}，它既标出晶格原点，也指定“第 0 步使用哪个模板”。
#    - 其余单点都用颜色 O。把单点相对原点的偏移除以统一间距 gunit 后，按 (dr+dc) 的奇偶交替选模板：
#      偶数层用起始模板，奇数层用另一个模板。原点那份仍重着色为 O，若发生覆盖重叠也着色为 O。
#    - 小例子：若原题在 deltas={(0,0),(0,1),(1,1)} 时会输出 T,T,T 三份（原点改 O），
#      变体则会输出 A,B,A 三份（若 origin 选中 A）；差异发生在“每个偏移点用哪一个模板”这一步。
# b) 复杂度校准:
#    - 只引入一个新的自由度：奇偶层二选一，不引入旋转/镜像/排序等额外机制。
#    - 两模板在每个样例里都会重复使用，因为 4 连通晶格中只要有多于 1 个点，就必然同时出现奇偶两类位置。
# c) 隐式假设审查:
#    - 原版 `max(comps, key=area)` 依赖“唯一大模板”；现在必须找出两个非单例模板。
#    - 原版 `templ_col` 同时代表唯一模板颜色与原点颜色；现在原点颜色只决定起始模板，不决定所有 stamp 的颜色。
#    - 原版 `templ_rel` 可无条件复用到全部位置；现在必须按奇偶层在 `templ_rel_a/templ_rel_b` 之间切换。
#    - 原版 `out[total>0] = templ_col` 默认所有覆盖像素都同色；现在必须保留逐像素的模板颜色，不能覆盖掉 B 模板。
# d) 放置合法性:
#    - 生成器强制两个模板拥有相同 bbox 尺寸，这样仍可沿用原题的 `(dr*h, dc*w)` 平移缩放规则。
#    - 起始模板放在输出锚点；另一个模板仅作为“备用原型”单独摆放。单点晶格与备用模板都与输入中的可见对象保持 >=1 背景间隔。
#    - 所有未来 stamp bbox 都被约束在 22x22 内；生成器还验证 `main(generate_input())` 不报错。
# e) 视觉可观察性:
#    - A/B 模板颜色不同且形状不同，解题者能直接看到“有两个原型”，并从输出中看出两者交替出现。
#    - 交替依据是晶格上的空间奇偶层，不依赖颜色 id 大小、扫描顺序等不可见因素。
# f) 可视化整洁度:
#    - 输入中只保留三块区域：起始模板、备用模板、单点晶格；三者彼此分离，避免把多个线索堆在同一区域。
#    - 输出仍主要是一组平移 stamp，视觉结构和原题接近，只是在晶格上出现 A/B 交替。
# g) 颜色变化可推断性:
#    - 颜色总数固定为 4：bg、A、B、O。A/B 只编码两个模板，O 只编码单点配色与重叠/原点改色。
#    - 没有额外的颜色映射表；颜色变化完全由“是否原点/是否重叠/当前奇偶层选哪块模板”决定。
# 9) main() 的新增输入假设:
#    - 恰有两个非单例单色连通块，且 bbox 尺寸相同。
#    - 恰有一个单点颜色属于两个模板颜色之一；它指定起始模板。
#    - 晶格点数 >= 2，因此奇偶层都实际出现。`generate_input()` 明确保证这些前提。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp alternating copies of two monochrome templates guided by a singleton lattice.

    Input structure:
    - bg = background. Two disconnected monochrome template sprites appear in colors A and B, sharing one bbox size but using different shapes.
    - Elsewhere, singleton pixels form a uniform-spacing lattice: exactly one singleton uses color A or B, and all remaining singletons use color O.
    - The unique A/B-colored singleton marks the lattice origin and selects which template is the step-0 anchor.

    Rule:
    - Convert singleton offsets from the origin into coarse lattice units using the common singleton spacing.
    - Stamp translated copies from the selected anchor template position, using the start template on even `(dr + dc)` parity and the other template on odd parity.
    - Recolor the origin copy to O, and recolor any cell covered by two or more stamped copies to O as well.

    Key insight: The colored origin singleton carries two pieces of information at once: it chooses the anchor template in the input and fixes the alternating parity pattern across the whole lattice.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps

    templates = [comp for comp in comps if int(comp["area"]) > 1]
    assert len(templates) == 2

    template_info = []
    for comp in templates:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = int(bbox_h((r0, c0, r1, c1)))
        w = int(bbox_w((r0, c0, r1, c1)))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        template_info.append(
            {
                "color": int(comp["color"]),
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
                "rel": rel,
            }
        )

    dims = {(info["h"], info["w"]) for info in template_info}
    assert len(dims) == 1
    h, w = next(iter(dims))
    template_colors = {info["color"] for info in template_info}

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((rr, cc, int(comp["color"])))
    assert len(singletons) >= 2

    origin = [(r, c, col) for (r, c, col) in singletons if col in template_colors]
    assert len(origin) == 1
    orr, orc, start_col = origin[0]

    other_cols = [col for (_r, _c, col) in singletons if col not in template_colors]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    start_template = next(info for info in template_info if info["color"] == start_col)
    alt_template = next(info for info in template_info if info["color"] != start_col)

    deltas = [(int(r - orr), int(c - orc)) for (r, c, _col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    coarse = [(int(dr // gunit), int(dc // gunit)) for dr, dc in deltas]

    total = np.zeros((H, W), dtype=int)
    origin_cover = np.zeros((H, W), dtype=int)
    paint = np.full((H, W), int(bg), dtype=int)

    for dr, dc in coarse:
        use_start = ((int(dr) + int(dc)) & 1) == 0
        templ = start_template if use_start else alt_template
        off_r = int(dr * h)
        off_c = int(dc * w)
        is_origin = (dr == 0 and dc == 0)
        for rr, cc in templ["rel"]:
            R = int(start_template["r0"] + off_r + rr)
            C = int(start_template["c0"] + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                if total[R, C] == 0:
                    paint[R, C] = int(templ["color"])
                total[R, C] += 1
                if is_origin:
                    origin_cover[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    single_mask = total == 1
    out[single_mask] = paint[single_mask]
    out[single_mask & (origin_cover > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


# --------------- generator (D12 variant) ---------------

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
PAIR_RETRIES = 80
SINGLETON_RETRIES = 60
PLACEMENT_RETRIES = 80
GENERATE_RETRIES = 400


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


def _canon(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in cells))


def _gen_template_pair(h: int, w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(PAIR_RETRIES):
        a = _gen_template(h, w)
        b = _gen_template(h, w)
        if a is None or b is None:
            continue
        if _canon(a) == _canon(b):
            continue
        if len(set(_canon(a)) ^ set(_canon(b))) < 2:
            continue
        return a, b
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


def _bboxes_separated(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int], gap: int = 1) -> bool:
    r0a, c0a, r1a, c1a = map(int, b1)
    r0b, c0b, r1b, c1b = map(int, b2)
    return (
        r1a + gap < r0b
        or r1b + gap < r0a
        or c1a + gap < c0b
        or c1b + gap < c0a
    )


def _place_singletons(
    deltas: list[tuple[int, int]],
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
    anchor_bbox: tuple[int, int, int, int],
) -> tuple[int, int, tuple[int, int, int, int]] | None:
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)
        cluster_bbox = (
            orr + min_dr * GUNIT,
            orc + min_dc * GUNIT,
            orr + max_dr * GUNIT,
            orc + max_dc * GUNIT,
        )
        if not _bboxes_separated(cluster_bbox, anchor_bbox, gap=1):
            continue
        return orr, orc, cluster_bbox
    return None


def _place_backup_template(
    h: int,
    w: int,
    anchor_bbox: tuple[int, int, int, int],
    cluster_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(0, GRID_SIZE - h)
        c0 = random.randint(0, GRID_SIZE - w)
        cand = (r0, c0, r0 + h - 1, c0 + w - 1)
        if not _bboxes_separated(cand, anchor_bbox, gap=1):
            continue
        if not _bboxes_separated(cand, cluster_bbox, gap=1):
            continue
        return r0, c0
    return None


def _gen_attempt() -> np.ndarray | None:
    bg, col_a, col_b, overlap_col = random.sample(range(10), 4)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    pair = _gen_template_pair(h, w)
    if pair is None:
        return None
    cells_a, cells_b = pair

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing)
    if lat is None:
        return None
    lattice_cells, origin = lat
    deltas = [(r - origin[0], c - origin[1]) for r, c in lattice_cells]
    if not any(((dr + dc) & 1) for dr, dc in deltas):
        return None
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    r0_lo, r0_hi = -min_dr * h, GRID_SIZE - max_dr * h - h
    c0_lo, c0_hi = -min_dc * w, GRID_SIZE - max_dc * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    anchor_r0 = random.randint(r0_lo, r0_hi)
    anchor_c0 = random.randint(c0_lo, c0_hi)
    anchor_bbox = (anchor_r0, anchor_c0, anchor_r0 + h - 1, anchor_c0 + w - 1)

    start_idx = random.randint(0, 1)
    start_cells = cells_a if start_idx == 0 else cells_b
    start_col = col_a if start_idx == 0 else col_b
    backup_cells = cells_b if start_idx == 0 else cells_a
    backup_col = col_b if start_idx == 0 else col_a

    singleton_data = _place_singletons(deltas, min_dr, max_dr, min_dc, max_dc, anchor_bbox)
    if singleton_data is None:
        return None
    orr, orc, cluster_bbox = singleton_data

    backup_pos = _place_backup_template(h, w, anchor_bbox, cluster_bbox)
    if backup_pos is None:
        return None
    backup_r0, backup_c0 = backup_pos

    grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    for tr, tc in start_cells:
        grid[anchor_r0 + tr, anchor_c0 + tc] = start_col
    for tr, tc in backup_cells:
        grid[backup_r0 + tr, backup_c0 + tc] = backup_col
    for dr, dc in deltas:
        rr = orr + dr * GUNIT
        cc = orc + dc * GUNIT
        grid[rr, cc] = start_col if (dr == 0 and dc == 0) else overlap_col

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
