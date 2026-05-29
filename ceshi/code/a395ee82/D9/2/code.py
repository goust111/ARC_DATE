# Variant of a395ee82
# Dimension: D9 — 输入中单例像素的保留/移除 (trajectory)
# Change: 单例像素在输出中被移除(消失)，模板被原点stamp覆盖(重着色为O)... → 原始模板保留

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis
# ---------------
# variant_spec:
#   source_task: none
#   core_change:
#     原题会把 singleton 晶格中的 T 色原点也解释成一个 stamp，因而原模板所在位置会被“自我副本”
#     重新着成 O。这个变体保留输入里的原模板: 输出中它仍是 T 色，只在其余 singleton 对应的偏移
#     位置绘制新副本。singletons 仍全部消失。
#   concrete_behavior:
#     输入仍由三部分组成:
#     1. 背景 bg
#     2. 一个最大的单色模板连通块，颜色 T
#     3. 5-6 个按等距晶格摆放的单像素，其中唯一一个 T 色像素是原点，其余都是 O 色
#     输出中保留原模板，并按所有非原点 singleton 的 (dr, dc) 偏移，在
#     (r0 + dr*h, c0 + dc*w) 处复制模板。新副本之间若发生覆盖，则覆盖处为 O；
#     但原模板区域是受保护区，不会被任何 stamp 改色。
#   behavior_trace:
#     若模板 bbox 为 3x3，原点在 singleton 晶格中心，另有偏移 (0,1) 与 (1,0) 两个 O 色单像素:
#     原逻辑: 在模板原位画一个 O 色副本，同时在右侧和下方各画一个 T 色副本。
#     变体逻辑: 原位完全不画，只保留输入中已有的 T 色模板；右侧和下方仍各画一个 T 色副本。
#     差异集中在“原位是否被 O 覆盖”，因此样例里肉眼可直接看到模板是否被保留。
#   complexity_calibration:
#     不增加新的对象类型、颜色类别或多档规则，仍保持 3 色、单模板、单晶格。
#     可观察差异稳定而不隐蔽: 原模板面积为 6-10，颜色 T 与 O 恒不同，因此输出里至少有 6 个格子
#     会与原题不同，不会退化成难以察觉的微小变化。
#   implicit_assumptions_review:
#     - `template = max(comps, key=area)`: 仍依赖模板是最大连通块。生成器保持其他对象都是 singleton，
#       且模板面积 6-10 > 1，因此成立。
#     - `origin` 由唯一 T 色 singleton 决定: 变体不改变这一识别方式，仍成立。
#     - 原题使用 `center` 统计原点 stamp 覆盖并把该区域改成 O；变体中这个前提失效，
#       因为“原点 singleton 必须被绘制”为旧属性。现在必须跳过原点 stamp，不能再无条件覆写。
#     - 输出若从空白网格直接画所有 stamp，会错误地把原模板也当普通副本处理。变体需要显式先保留
#       原模板，并把其区域作为不可覆盖 mask。
#   placement_legality:
#     原生成器已经保证所有按 (dr*h, dc*w) 平移得到的模板副本彼此不重叠、都在 22x22 内。
#     这比变体需要的条件更强，因此继续沿用即可；原模板自然也不会被其他副本撞上。
#   visual_observability:
#     解题者在 2-4 组样例里可以看到: singleton 全消失，但输入里的大模板不再变色，而是作为一个
#     特殊的“起点实例”被保留，其他位置才新增副本。规则单一、可重复、可直接比较。
#   visual_cleanliness:
#     不引入额外标记；输出仍只由整齐分布的模板副本组成。由于延续原生成器的不重叠约束，网格不会
#     出现难以辨认的多层覆盖。
#   color_inference:
#     颜色角色保持原题的结构线索: T 是模板/普通副本色，O 是 singleton 非原点色，也是副本重叠色。
#     本变体不新增颜色映射，只撤销原位改色，因此人类可从位置关系推断而不是依赖颜色 id。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a monochrome template using a singleton lattice while preserving the original template.

    Input structure:
    - bg = background. One large monochrome component is the template, colored T.
    - Separate singleton markers form a uniformly spaced lattice: exactly one singleton is color T and marks the lattice origin.
    - All remaining singleton markers share color O and indicate where translated copies should be drawn.

    Rule:
    - Read singleton offsets relative to the unique T-colored origin and normalize them by the lattice spacing.
    - Scale each offset by the template bbox height/width to locate copy positions.
    - Keep the input template at its original location unchanged; do not draw the origin's self-copy over it.
    - Draw copies only for the non-origin singleton offsets, and recolor only true overlaps between those new copies to O.

    Key insight: The T-colored singleton still defines the translation origin, but in this variant it is only a reference point; the template already present in the input is a protected instance rather than a stamp target.
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
    protected = np.zeros((H, W), dtype=bool)
    protected[templ_coords[:, 0], templ_coords[:, 1]] = True

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
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    total = np.zeros((H, W), dtype=int)
    for dr, dc, col in deltas:
        if int(col) == int(templ_col):
            continue
        off_r = int(dr * h)
        off_c = int(dc * w)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W and not protected[R, C]:
                total[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[templ_coords[:, 0], templ_coords[:, 1]] = int(templ_col)
    new_copy = (total > 0) & (~protected)
    overlap = (total >= 2) & (~protected)
    out[new_copy] = int(templ_col)
    out[overlap] = int(overlap_col)
    return out


# --------------- generator (a1, D9 variant) ---------------

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]         # template bbox height options
TEMPLATE_W = 3                      # template bbox width (fixed)
TEMPLATE_AREA_RANGE = (6, 10)       # absolute cell-count bounds
TEMPLATE_FILL_RANGE = (0.65, 0.85)  # fill-ratio bounds within bbox
SINGLETON_COUNT_RANGE = (5, 6)      # number of lattice singletons
LATTICE_EXTENT_RANGE = (3, 4)       # rows/cols of lattice polyomino
GUNIT = 2                           # singleton spacing in pixels

# Retry budgets
SPANNING_RETRIES = 40               # attempts per spanning polyomino
TEMPLATE_RETRIES = 60               # attempts to build a valid template
SINGLETON_RETRIES = 50              # attempts to place singleton cluster
GENERATE_RETRIES = 300              # outer retry loop


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
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
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


def _delta_bounds(deltas: list[tuple[int, int]]):
    """Return (min_dr, max_dr, min_dc, max_dc) for a list of (dr, dc) offsets."""
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _place_singletons(
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
    templ_col,
    overlap_col,
    occupied,
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
            grid[r0 + tr, c0 + tc] = templ_col
        for dr, dc in deltas:
            grid[orr + dr * GUNIT, orc + dc * GUNIT] = templ_col if (dr == 0 and dc == 0) else overlap_col

        try:
            main(grid)
        except Exception:
            return None
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, place stamps + singletons."""
    bg, templ_col, overlap_col = random.sample(range(10), 3)

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

    occupied = set()
    for dr, dc in deltas:
        for tr, tc in templ_cells:
            pt = (r0 + dr * h + tr, c0 + dc * w + tc)
            if pt in occupied:
                return None
            occupied.add(pt)

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
        templ_col,
        overlap_col,
        occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
