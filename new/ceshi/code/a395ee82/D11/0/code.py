# Variant of a395ee82
# Dimension: D11 — stamp放置基准点 (meta)
# Change: stamp以模板外接矩形左上角为基准点，按晶格偏移×模板尺寸(h,w)进行平移放置... → 以模板重心为基准点

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis
# variant:
#   dimension: D11
#   source_task: none
#   concrete_behavior:
#     input仍然只有两个对象族: 一个单色模板(T)和一组单例像素晶格(1个T色原点 + 若干O色点)。
#     变化发生在output里“复制模板”的锚点解释:
#       old: 把模板bbox左上角当成放置基准，副本左上角 = (r0, c0) + (dr*h, dc*w)。
#       new: 仍然先得到同样的晶格锚点 (r0 + dr*h, c0 + dc*w)，但这个锚点现在对应模板像素重心，
#            因而副本左上角 = round((r0 + dr*h, c0 + dc*w) - centroid(template_pixels))。
#     这样origin副本通常不再盖回输入模板本体，而会整体向左上偏移1格或2格；不同模板的重心不同，偏移也不同。
#   behavior_trace:
#     例: h=4,w=3，模板像素重心(local)=(1.67, 1.00)，某晶格偏移为(dr,dc)=(1,-1)。
#       old副本左上角 = (r0+4, c0-3)。
#       new副本左上角 = (round(r0+4-1.67), round(c0-3-1.00)) = (r0+2, c0-4)。
#     差异不是颜色或对象数量变化，而是同一批stamp整体以“重心对晶格”方式落位。
#   complexity_calibration:
#     保留原任务的三色、单模板、单例晶格和重叠着色机制，避免引入额外映射自由度。
#     为了让D11可观察，generator会拒绝那些在同一输入上 new_output == old_output 的退化样本。
#   implicit_assumptions_review:
#     template = max(comps, key=area):
#       仍假设输入里模板是最大连通块；generator继续只放1个大模板 + 若干单例，保证成立。
#     templ_rel = coords - bbox_topleft:
#       旧版默认“局部坐标直接加到左上角平移量”；新版仍复用局部坐标，但必须先减去重心再取整生成新的左上角。
#     off_r = dr * h / off_c = dc * w:
#       这些量在新版不再是最终副本左上角，只是“晶格锚点”的整数位移。
#     out[(total == 1) & (center > 0)] = overlap_col:
#       仍依赖“唯一T色单例对应origin副本”；规则未变，继续成立。
#   placement_legality:
#     generator先枚举可用(r0,c0)，要求按新锚点计算的所有副本:
#       1. 全部在22x22内；
#       2. 彼此不重叠；
#       3. 单例晶格不落在任何未来stamp像素上；
#       4. 输入模板bbox与单例晶格bbox仍至少隔1个背景像素。
#   observability:
#     人类仍能先从输入识别“模板 + 晶格”，再从输出看出副本按晶格扩张。
#     与原题不同的是，副本阵列相对输入模板整体挪动到了“重心对齐”的位置，而不是从模板左上角整齐平铺。
#   visual_cleanliness:
#     保持原题的稀疏布局与少量颜色，不增加覆盖层次；重叠仍只有O色一种提示。
#   color_inference:
#     颜色规则不变: T是常规stamp色，origin副本与所有重叠区域改成O；没有使用任何不可见颜色id排序。


def _round_nearest(x: float) -> int:
    """Round to the nearest grid coordinate with half-up behavior on positive values."""
    return int(math.floor(float(x) + 0.5))


def _template_centroid(templ_rel: list[tuple[int, int]]) -> tuple[float, float]:
    rs = [r for r, _c in templ_rel]
    cs = [c for _r, c in templ_rel]
    return float(sum(rs) / len(rs)), float(sum(cs) / len(cs))


def _stamp_shift(
    r0: int,
    c0: int,
    dr: int,
    dc: int,
    h: int,
    w: int,
    centroid_r: float,
    centroid_c: float,
    *,
    anchor_mode: str,
) -> tuple[int, int]:
    anchor_r = int(r0 + dr * h)
    anchor_c = int(c0 + dc * w)
    if anchor_mode == "bbox":
        return anchor_r, anchor_c
    if anchor_mode == "centroid":
        return _round_nearest(anchor_r - centroid_r), _round_nearest(anchor_c - centroid_c)
    raise ValueError(f"Unknown anchor_mode: {anchor_mode}")


def _solve(grid: np.ndarray, *, anchor_mode: str) -> np.ndarray:
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
    centroid_r, centroid_c = _template_centroid(templ_rel)

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = int(math.gcd(*gcd_vals))
    gunit = max(1, int(gunit))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc, col in deltas:
        shift_r, shift_c = _stamp_shift(
            r0, c0, dr, dc, h, w, centroid_r, centroid_c, anchor_mode=anchor_mode
        )
        is_origin = int(col) == int(templ_col)
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
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a monochrome template using a singleton lattice, but anchor each stamp at the template centroid.

    Input structure:
    - A single large monochrome template component in color T sits on the background.
    - Elsewhere, singleton pixels form a uniformly spaced lattice: exactly one singleton is color T and the others are color O.
    - The T-colored singleton marks the lattice origin; O is also the recolor used for the origin copy and any overlaps.

    Rule:
    - Recover lattice offsets by measuring singleton displacements from the unique T-colored origin and dividing by the lattice spacing.
    - Convert each lattice offset into an integer anchor step of (dr*h, dc*w), where h and w come from the template bbox.
    - Treat those stepped positions as target locations for the template centroid, then round back to integer grid cells to place each whole stamp.
    - Paint the origin copy in O instead of T, and repaint any pixel covered by multiple copies in O as well.

    Key insight:
    - The singleton lattice still tells you which copies to create, but the geometric reference point is no longer the template's top-left corner; it is the template's pixel centroid.
    """
    return _solve(grid, anchor_mode="centroid")


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
SINGLETON_RETRIES = 50
GENERATE_RETRIES = 300


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of *n* cells in a rows x cols grid."""
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
    """Polyomino of *n* cells that spans exactly rows x cols (touches all 4 edges)."""
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _gen_template(h: int, w: int) -> list[tuple[int, int]] | None:
    """4-connected template shape in h x w, spanning bbox, within area/fill bounds."""
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
    """Lattice polyomino of *n* cells with strictly interior origin."""
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


def _stamp_pixels(
    deltas: list[tuple[int, int]],
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], tuple[float, float]]:
    centroid_r, centroid_c = _template_centroid(templ_cells)
    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        shift_r, shift_c = _stamp_shift(
            r0, c0, dr, dc, h, w, centroid_r, centroid_c, anchor_mode="centroid"
        )
        for tr, tc in templ_cells:
            occupied.add((shift_r + tr, shift_c + tc))
    return occupied, (centroid_r, centroid_c)


def _valid_template_origins(
    deltas: list[tuple[int, int]],
    h: int,
    w: int,
    centroid_r: float,
    centroid_c: float,
) -> tuple[list[int], list[int]]:
    valid_r0: list[int] = []
    valid_c0: list[int] = []

    for r0 in range(GRID_SIZE - h + 1):
        ok = True
        for dr, _dc in deltas:
            shift_r = _round_nearest(r0 + dr * h - centroid_r)
            if shift_r < 0 or shift_r + h - 1 >= GRID_SIZE:
                ok = False
                break
        if ok:
            valid_r0.append(r0)

    for c0 in range(GRID_SIZE - w + 1):
        ok = True
        for _dr, dc in deltas:
            shift_c = _round_nearest(c0 + dc * w - centroid_c)
            if shift_c < 0 or shift_c + w - 1 >= GRID_SIZE:
                ok = False
                break
        if ok:
            valid_c0.append(c0)

    return valid_r0, valid_c0


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
    """Try to find a valid singleton cluster position and build the input grid."""
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
            grid[r0 + tr, c0 + tc] = T
        for dr, dc in deltas:
            grid[orr + dr * GUNIT, orc + dc * GUNIT] = T if (dr == 0 and dc == 0) else O

        try:
            new_out = main(grid)
            old_out = _solve(grid, anchor_mode="bbox")
        except Exception:
            return None

        diff = int(np.count_nonzero(new_out != old_out))
        if diff < max(4, len(templ_cells)):
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, and place the new centroid-anchored stamps."""
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

    centroid_r, centroid_c = _template_centroid(templ_cells)
    valid_r0, valid_c0 = _valid_template_origins(deltas, h, w, centroid_r, centroid_c)
    if not valid_r0 or not valid_c0:
        return None

    r0 = random.choice(valid_r0)
    c0 = random.choice(valid_c0)

    occupied, _ = _stamp_pixels(deltas, templ_cells, r0, c0, h, w)
    if len(occupied) != len(templ_cells) * len(deltas):
        return None

    if any(not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE) for r, c in occupied):
        return None

    return _place_singletons(
        deltas,
        min_dr,
        max_dr,
        min_dc,
        max_dc,
        templ_cells,
        r0,
        c0,
        h,
        w,
        bg,
        T,
        O,
        occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
