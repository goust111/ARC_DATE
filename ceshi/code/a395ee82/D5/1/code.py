# Variant of a395ee82
# Dimension: D5 — stamp重叠处理规则 (transform)
# Change: 重叠像素(被两个或以上stamp覆盖)统一着O色... → 重叠像素消失
#
# design_analysis:
#   concretization:
#     - 输入仍由两个部分组成: 一个单色模板精灵(T色) + 一组按等间距摆放的单例像素。
#     - 单例像素中恰有一个也是T色, 它给出原点; 其他单例统一是O色。
#     - 输出仍是按单例晶格去复制模板, 但为了让 D5 可观察, 相邻副本不再用整块 bbox
#       尺寸作步长, 而是用 (h-1, w-1) 作步长, 这样相邻副本会共享一条边带。
#     - 新规则下, 任意被两个或以上副本同时覆盖的像素都会直接变回背景色, 形成缺口/
#       干涉带; 只有“恰好被原点副本单独覆盖”的像素才会着 O,色。
#     - 小例子: 若模板高 4 宽 3, 两个晶格点在竖直方向相邻, 原题会把第二份模板放在
#       下移 4 行的位置, 两份只接边不重叠; 变体改为下移 3 行, 两份在一行上重叠。
#       原题若该行重叠会整行涂 O; 变体则把这些重叠像素擦成背景。
#   complexity_calibration:
#     - 若仍使用原题步长 (h, w), D5 改动在绝大多数样例中不可见, 会退化回原题。
#     - 因此仅做最小配套改动: 缩短平移步长 1 格, 其余识别机制、颜色角色、晶格来源、
#       原点编码都保持不变。
#     - generator 强制样例至少出现若干个 cancel 像素, 同时保留可见的 T 区与 O 区,
#       防止输出退化成纯背景或只有单一颜色。
#   implicit_assumption_review:
#     - 原代码 `off_r = dr * h`, `off_c = dc * w` 依赖“bbox 刚好不重叠”; 变体不再成立,
#       必须改为 `(h-1, w-1)` 才能产生 D5 所需的重叠。
#     - 原代码 `out[total >= 2] = overlap_col` 直接把重叠涂 O; 新规则下这里必须删除,
#       否则会保留旧行为。
#     - 原 generator 的 `occupied` 检查依赖“输出副本绝不重叠”; 变体里这条假设错误,
#       应改为统计覆盖次数, 接受重叠并要求其可见。
#   placement_legality:
#     - 所有 stamp 的放置窗口按新步长计算, 保证完整落在 22x22 内。
#     - 输入里的模板与单例簇仍保持 bbox 分离且至少隔 1 个背景格。
#     - 单例像素仍不落在输出 footprint 上, 让输入结构整洁、组件解析稳定。
#   observability:
#     - 人类可从 2-4 组样例看到: 相邻副本并非简单并排, 而是互相“咬掉”一条带;
#       被咬掉的部分都变成背景而不是 O 色。
#     - 颜色逻辑只保留三色角色 bg/T/O, 不做任何基于颜色 id 的隐藏排序。
#   cleanliness:
#     - 通过限制模板尺寸、晶格规模与最小 cancel 数, 输出会出现少量清晰缺口,
#       不会大面积乱叠。
#   new_main_assumptions:
#     - `main()` 现在假设至少有一个 T 色模板组件、唯一的 T 色原点单例、以及统一间距
#       的单例晶格; generator 按原题同样的识别线索保证这些条件。
#     - `main()` 还假设新步长下至少有部分原点像素未被别的副本抵消, 否则 O 色会消失;
#       generator 明确检查 `origin_visible > 0`。

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


def _stamp_steps(h: int, w: int) -> tuple[int, int]:
    """Stride is reduced by one cell per axis so neighboring copies can interfere."""
    return max(1, int(h) - 1), max(1, int(w) - 1)


def _coverage_maps(
    shape: tuple[int, int],
    base_r: int,
    base_c: int,
    templ_rel: list[tuple[int, int]],
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return total coverage count and coverage contributed by the origin copy."""
    H, W = map(int, shape)
    step_r, step_c = _stamp_steps(h, w)
    total = np.zeros((H, W), dtype=int)
    origin = np.zeros((H, W), dtype=int)
    for dr, dc in deltas:
        off_r = int(dr * step_r)
        off_c = int(dc * step_c)
        is_origin = int(dr) == 0 and int(dc) == 0
        for rr, cc in templ_rel:
            R = int(base_r + off_r + rr)
            C = int(base_c + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    origin[R, C] += 1
    return total, origin


def _render_output(
    bg: int,
    templ_col: int,
    overlap_col: int,
    total: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Render the variant output from precomputed coverage maps."""
    out = np.full(total.shape, int(bg), dtype=int)
    out[(total == 1) & (origin == 0)] = int(templ_col)
    out[(total == 1) & (origin > 0)] = int(overlap_col)
    return out


def _variant_stats(
    shape: tuple[int, int],
    base_r: int,
    base_c: int,
    templ_rel: list[tuple[int, int]],
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
) -> dict[str, int]:
    total, origin = _coverage_maps(shape, base_r, base_c, templ_rel, h, w, deltas)
    return {
        "cancel_cells": int(np.sum(total >= 2)),
        "origin_visible": int(np.sum((total == 1) & (origin > 0))),
        "templ_visible": int(np.sum((total == 1) & (origin == 0))),
        "footprint_cells": int(np.sum(total > 0)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated template copies from a singleton lattice, with interference erasing overlaps.

    Input structure:
    - bg is the dominant background color. One large monochrome component is the template in color T.
    - Elsewhere lie singleton lattice markers: exactly one singleton is also T and marks the lattice origin.
    - All remaining singletons share color O and sit on a uniform coarse grid around that origin.

    Rule:
    - Recover each singleton's lattice offset relative to the unique T singleton.
    - Copy the template to every offset, but use a stride of (template height - 1, template width - 1),
      so neighboring copies can overlap.
    - Pixels covered only by the origin copy become O; pixels covered only by non-origin copies stay T.
    - Pixels covered by two or more copies disappear back to the background color.

    Key insight: The singleton lattice still encodes where copies go, but now the visible clue is destructive
    overlap: shared stamp cells cancel out instead of turning into the overlap color.
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
        if comp is template or int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas_pix = [(int(r - orr), int(c - orc)) for (r, c, _col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc in deltas_pix for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit)) for dr, dc in deltas_pix]

    total, origin_cover = _coverage_maps((H, W), r0, c0, templ_rel, h, w, deltas)
    return _render_output(bg, templ_col, overlap_col, total, origin_cover)


# --------------- generator (a1) ---------------

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2
MIN_CANCEL_CELLS = 2
MIN_ORIGIN_VISIBLE = 2
MIN_TEMPLATE_VISIBLE = 4

# Retry budgets
SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
SINGLETON_RETRIES = 50
GENERATE_RETRIES = 800


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


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return (min_dr, max_dr, min_dc, max_dc) for a list of (dr, dc) offsets."""
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _footprint_points(
    base_r: int,
    base_c: int,
    templ_cells: list[tuple[int, int]],
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    """All cells touched by at least one output stamp."""
    step_r, step_c = _stamp_steps(h, w)
    pts: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        for tr, tc in templ_cells:
            pts.add((int(base_r + dr * step_r + tr), int(base_c + dc * step_c + tc)))
    return pts


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

        total, origin = _coverage_maps((GRID_SIZE, GRID_SIZE), r0, c0, templ_cells, h, w, deltas)
        expected = _render_output(bg, templ_col, overlap_col, total, origin)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return None
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, place a visibly interfering layout."""
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

    step_r, step_c = _stamp_steps(h, w)
    r0_lo, r0_hi = -min_dr * step_r, GRID_SIZE - max_dr * step_r - h
    c0_lo, c0_hi = -min_dc * step_c, GRID_SIZE - max_dc * step_c - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)

    stats = _variant_stats((GRID_SIZE, GRID_SIZE), r0, c0, templ_cells, h, w, deltas)
    if stats["cancel_cells"] < MIN_CANCEL_CELLS:
        return None
    if stats["origin_visible"] < MIN_ORIGIN_VISIBLE:
        return None
    if stats["templ_visible"] < MIN_TEMPLATE_VISIBLE:
        return None

    occupied = _footprint_points(r0, c0, templ_cells, h, w, deltas)
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
