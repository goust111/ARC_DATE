# Combo variant of a395ee82 (index 19)
# combo_spec: D12/0+D13/0
# Name: 双模板交替四色

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis
# a) 具体化与行为追踪:
#    - 融合后输入里有两个分离的单色模板块，bbox 尺寸相同但形状不同。除此之外，还有一组等间距单点晶格。
#    - 由于组合描述要求“四色”，而 D12 的双模板 + D13 的独立原点标记若直接相加会变成 5 色，
#      这里采用最小冲突的整合：保留唯一原点标记色 M；把原题里的重着色 O 合并到模板 B 的颜色上。
#      于是总颜色为 bg、A、B(同时也是 O)、M 四种。
#    - 小例子：若粗晶格偏移为 {(0,0),(0,1),(1,0),(1,1),(2,1)}，则偶奇性分别为
#      even, odd, odd, even, odd。融合后输出在 even 位置盖 A 形，在 odd 位置盖 B 形；
#      其中 (0,0) 的 A 形整块改涂成 B 色。与原题相比，变化发生在两处：
#      1) 原点不再靠“单点颜色=模板色”识别，而靠唯一的 M 识别；
#      2) 输出不再是单模板重复，而是 A/B 交替。
# b) 复杂度校准:
#    - 只引入两个稳定自由度：模板二选一、晶格奇偶二分。没有旋转、镜像、排序、颜色表等额外机制。
#    - 生成器强制 odd 位置至少 2 个、非原点 even 位置至少 2 个，避免“只看见一种模板颜色”或某一支只出现一次而退化。
# c) 隐式假设审查:
#    - 原版 `max(comps, key=area)` 假定只有一个大模板；这里必须改成“恰有两个非单例单色连通块”。
#    - 原版用 `templ_col` 在单点中找原点；这里失效，因为原点只由 M 标出。
#    - 原版所有 stamp 共用同一 `templ_rel`；这里要按 `(dr + dc) % 2` 在 A/B 两套相对坐标间切换。
#    - 原版 `out[total > 0] = templ_col` 默认所有覆盖像素同色；这里必须逐像素记录第一次覆盖对应的模板颜色。
#    - 新增的一条融合假设是：非原点单点的重复颜色就是模板 B 的颜色，因此它同时承担 odd 模板色和重着色 O。
# d) 放置合法性:
#    - 两模板 bbox 始终相同，因此仍能沿用 `(dr*h, dc*w)` 的平移规则。
#    - 所有未来 stamp 都先按输出坐标系验证在界内且互不重叠；单点晶格不能落在任何未来 stamp 像素上。
#    - 输入里的备用模板必须与锚模板、单点晶格、以及所有未来 stamp bbox 保持至少 1 格背景间隔，避免视觉混淆。
# e) 视觉可观察性:
#    - 解题者能直接看到两个不同形状模板，以及一个唯一颜色 M 标出的晶格原点。
#    - 非原点单点都与模板 B 同色，这给出“B 既是 odd 模板色也是重着色”的可见线索。
#    - 奇偶交替依据是空间上的棋盘奇偶 `(dr+dc)`，而不是颜色编号、扫描顺序等不可见因素。
# f) 可视化整洁度:
#    - 输入只保留三类对象：锚模板 A、备用模板 B、单点晶格；三者全部分区摆放。
#    - 生成器额外避免备用模板压进未来输出区域，所以输入不会出现“看起来像已经盖过章”的噪声。
# g) 变化的可推断性:
#    - 颜色总数固定为 4；其中 M 只负责唯一原点，A/B 负责两种模板，B 额外兼任 O。
#    - 因为 origin 副本会把 A 形改成 B 色，所以少量样例就能看出：B 不只是第二模板，也是一种特殊重着色结果。
# 9) main() 的新增输入假设:
#    - 恰有两个非单例单色连通块，且 bbox 尺寸相同、形状不同。
#    - 单点颜色恰好有两类：唯一的非模板色 M，与重复出现且等于模板 B 颜色的 B/O 色。
#    - 单点晶格至少包含 2 个 odd 位置和 2 个非原点 even 位置。`generate_input()` 明确保证这些前提。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp alternating copies of two monochrome templates using a singleton lattice with a unique marker origin.

    Input structure:
    - `bg` is the background color. Two disconnected monochrome template sprites are visible; they share one bbox size but have different shapes.
    - Separate singleton pixels form a uniform-spacing lattice. Exactly one singleton uses a dedicated marker color `M`.
    - All other singleton pixels use the same color as one of the templates; that repeated color plays a dual role: it is both template `B` and the repaint color `O`.

    Rule:
    - Use the unique `M` singleton as the lattice origin, then convert all singleton offsets into coarse lattice units by dividing by the common spacing.
    - Take the template whose color does not appear among the repeated singleton pixels as template `A`, and the other one as template `B/O`.
    - Stamp translated copies from template `A`'s bbox origin: even `(dr + dc)` offsets use template `A`, odd offsets use template `B`.
    - Recolor the origin copy to `B/O`, and recolor any multiply covered cells to `B/O` as well.

    Key insight:
    - The solver must fuse shape, frequency, and color-role evidence: `M` identifies the origin, while the repeated singleton color identifies both the odd-phase template and the repaint color.
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
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        template_info.append(
            {
                "color": int(comp["color"]),
                "r0": r0,
                "c0": c0,
                "h": int(bbox_h((r0, c0, r1, c1))),
                "w": int(bbox_w((r0, c0, r1, c1))),
                "rel": rel,
                "shape": frozenset(rel),
            }
        )

    dims = {(info["h"], info["w"]) for info in template_info}
    assert len(dims) == 1
    h, w = next(iter(dims))
    assert len({info["shape"] for info in template_info}) == 2
    template_colors = {info["color"] for info in template_info}

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((rr, cc, int(comp["color"])))
    assert len(singletons) >= 3

    counts = Counter(col for _r, _c, col in singletons)
    marker_cols = [col for col, cnt in counts.items() if cnt == 1 and col not in template_colors]
    assert len(marker_cols) == 1
    marker_col = int(marker_cols[0])

    repaint_cols = [col for col in counts if col != marker_col]
    assert len(repaint_cols) == 1
    repaint_col = int(repaint_cols[0])
    assert repaint_col in template_colors

    origin = [(r, c) for (r, c, col) in singletons if col == marker_col]
    assert len(origin) == 1
    orr, orc = origin[0]

    start_template = next(info for info in template_info if info["color"] != repaint_col)
    alt_template = next(info for info in template_info if info["color"] == repaint_col)

    raw_deltas = [(int(r - orr), int(c - orc)) for (r, c, _col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc in raw_deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    coarse = [(int(dr // gunit), int(dc // gunit)) for dr, dc in raw_deltas]
    assert any(((dr + dc) & 1) for dr, dc in coarse)
    assert any((dr != 0 or dc != 0) and (((dr + dc) & 1) == 0) for dr, dc in coarse)

    total = np.zeros((H, W), dtype=int)
    origin_cover = np.zeros((H, W), dtype=int)
    paint = np.full((H, W), int(bg), dtype=int)

    for dr, dc in coarse:
        templ = start_template if (((dr + dc) & 1) == 0) else alt_template
        off_r = int(dr * h)
        off_c = int(dc * w)
        is_origin = dr == 0 and dc == 0
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
    out[single_mask & (origin_cover > 0)] = int(repaint_col)
    out[total >= 2] = int(repaint_col)
    return out


GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2
MIN_TEMPLATE_SYM_DIFF = 4
MIN_ODD_STAMPS = 2
MIN_EXTRA_EVEN_STAMPS = 2

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
PAIR_RETRIES = 80
LATTICE_RETRIES = 80
SINGLETON_RETRIES = 60
PLACEMENT_RETRIES = 120
GENERATE_RETRIES = 500


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
        cells_a = _gen_template(h, w)
        cells_b = _gen_template(h, w)
        if cells_a is None or cells_b is None:
            continue
        can_a = set(_canon(cells_a))
        can_b = set(_canon(cells_b))
        if can_a == can_b:
            continue
        # Keep the two visible template sprites clearly distinct for human solvers.
        if len(can_a ^ can_b) < MIN_TEMPLATE_SYM_DIFF:
            continue
        return cells_a, cells_b
    return None


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(LATTICE_RETRIES):
        ext_r = random.randint(*LATTICE_EXTENT_RANGE)
        ext_c = random.randint(*LATTICE_EXTENT_RANGE)
        cells = _spanning_polyomino(n, ext_r, ext_c)
        if cells is None:
            continue
        interior = [(r, c) for r, c in cells if 0 < r < ext_r - 1 and 0 < c < ext_c - 1]
        if not interior:
            continue
        origin = random.choice(interior)
        deltas = [(r - origin[0], c - origin[1]) for r, c in cells]
        odd_count = sum(((dr + dc) & 1) for dr, dc in deltas)
        extra_even = sum((dr != 0 or dc != 0) and (((dr + dc) & 1) == 0) for dr, dc in deltas)
        if odd_count >= MIN_ODD_STAMPS and extra_even >= MIN_EXTRA_EVEN_STAMPS:
            return cells, origin
    return None


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


def _stamp_layout(
    anchor_r0: int,
    anchor_c0: int,
    h: int,
    w: int,
    start_cells: list[tuple[int, int]],
    alt_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], list[tuple[int, int, int, int]]] | None:
    occupied: set[tuple[int, int]] = set()
    stamp_bboxes: list[tuple[int, int, int, int]] = []
    for dr, dc in deltas:
        base_r = anchor_r0 + dr * h
        base_c = anchor_c0 + dc * w
        stamp_bboxes.append((base_r, base_c, base_r + h - 1, base_c + w - 1))
        cells = start_cells if (((dr + dc) & 1) == 0) else alt_cells
        for tr, tc in cells:
            pt = (base_r + tr, base_c + tc)
            if pt in occupied:
                return None
            occupied.add(pt)
    return occupied, stamp_bboxes


def _place_singletons(
    deltas: list[tuple[int, int]],
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
    anchor_bbox: tuple[int, int, int, int],
    stamp_pixels: set[tuple[int, int]],
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
        if any((orr + dr * GUNIT, orc + dc * GUNIT) in stamp_pixels for dr, dc in deltas):
            continue
        return orr, orc, cluster_bbox
    return None


def _place_backup_template(
    h: int,
    w: int,
    anchor_bbox: tuple[int, int, int, int],
    cluster_bbox: tuple[int, int, int, int],
    stamp_bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(0, GRID_SIZE - h)
        c0 = random.randint(0, GRID_SIZE - w)
        cand = (r0, c0, r0 + h - 1, c0 + w - 1)
        if not _bboxes_separated(cand, anchor_bbox, gap=1):
            continue
        if not _bboxes_separated(cand, cluster_bbox, gap=1):
            continue
        if any(not _bboxes_separated(cand, bbox, gap=1) for bbox in stamp_bboxes):
            continue
        return r0, c0
    return None


def _gen_attempt() -> np.ndarray | None:
    bg, start_col, alt_col, marker_col = random.sample(range(10), 4)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    pair = _gen_template_pair(h, w)
    if pair is None:
        return None
    start_cells, alt_cells = pair

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lattice = _gen_lattice(n_sing)
    if lattice is None:
        return None
    lattice_cells, origin = lattice
    deltas = [(r - origin[0], c - origin[1]) for r, c in lattice_cells]
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    r0_lo, r0_hi = -min_dr * h, GRID_SIZE - max_dr * h - h
    c0_lo, c0_hi = -min_dc * w, GRID_SIZE - max_dc * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    anchor_r0 = random.randint(r0_lo, r0_hi)
    anchor_c0 = random.randint(c0_lo, c0_hi)
    anchor_bbox = (anchor_r0, anchor_c0, anchor_r0 + h - 1, anchor_c0 + w - 1)

    layout = _stamp_layout(anchor_r0, anchor_c0, h, w, start_cells, alt_cells, deltas)
    if layout is None:
        return None
    stamp_pixels, stamp_bboxes = layout

    singleton_data = _place_singletons(
        deltas,
        min_dr,
        max_dr,
        min_dc,
        max_dc,
        anchor_bbox,
        stamp_pixels,
    )
    if singleton_data is None:
        return None
    orr, orc, cluster_bbox = singleton_data

    backup_pos = _place_backup_template(h, w, anchor_bbox, cluster_bbox, stamp_bboxes)
    if backup_pos is None:
        return None
    backup_r0, backup_c0 = backup_pos

    grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    for tr, tc in start_cells:
        grid[anchor_r0 + tr, anchor_c0 + tc] = start_col
    for tr, tc in alt_cells:
        grid[backup_r0 + tr, backup_c0 + tc] = alt_col
    for dr, dc in deltas:
        rr = orr + dr * GUNIT
        cc = orc + dc * GUNIT
        grid[rr, cc] = marker_col if (dr == 0 and dc == 0) else alt_col

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.any(out == start_col) or not np.any(out == alt_col):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
