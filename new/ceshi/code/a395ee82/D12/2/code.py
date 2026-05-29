# Variant of a395ee82
# Dimension: D12 — 多步处理序列 (multi_step)
# Change: 单步处理 — 直接从输入计算输出(识别模板、解析晶格、stamp所有副本)... → 先stamp后连接

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D12 variant
# -----------------------------------
# a) 具体化与行为追踪
# spec 保留了原题输入: 一个单色模板 T, 以及远处一个由单例像素构成的 4 邻接晶格。
# 晶格里唯一的 T 色单例是原点, 其余单例都是 O 色。先按原题规则把模板复制到所有晶格偏移:
# 每个晶格单位位移 (dr, dc) 被放大成模板 bbox 位移 (dr*h, dc*w), 原点副本整体改成 O 色,
# 多重覆盖也改成 O 色。然后追加第二步: 只看晶格上的 4 邻接边, 若两个副本对应的晶格单元
# 相邻(差值恰为 (1,0) 或 (0,1)), 就在这两个副本之间画一条 O 色的正交连线。
# 为了让“连线”稳定且可从输入恢复, 连线锚点固定在每个副本 bbox 的中心行/中心列:
# 水平邻接时, 在共同的中心行上从左副本 bbox 左边界一直画到右副本 bbox 右边界;
# 垂直邻接时, 在共同的中心列上从上副本 bbox 上边界一直画到下副本 bbox 下边界。
# 例子: 若 h=4,w=3, 晶格有 (0,0),(0,1),(1,1), 原题输出会出现三个 stamp; 新变体会先得到
# 三个 stamp, 再在前两个 stamp 的中心横线处涂 O, 以及后两个 stamp 的中心竖线处涂 O。
#
# b) 复杂度校准
# 变体没有引入新的对象类型或颜色映射, 只增加第二个确定步骤, 所以不会让匹配歧义暴增。
# 但若 generator 总是只生成纯水平链或纯垂直链, 视觉差异会偏弱; 因此晶格额外约束为同时含有
# 至少一条水平邻接边和一条垂直邻接边, 让样例里反复出现“先复制, 再连成网”的效果。
#
# c) 隐式假设审查
# 原 main() 的关键前提包括:
# - 最大连通块就是模板, 该前提仍成立, 因为输入没有加入新的非单例对象。
# - 单例里恰有一个颜色等于模板色, 仍成立。
# - 晶格位移可由单例相对原点的 gcd 恢复, 仍成立。
# - 输出只依赖 stamp 覆盖计数。这里不再成立, 因为第二步还依赖“哪些晶格单元彼此 4 邻接”。
#   所以 solver 不能在完成第一步后直接返回, 还必须把规范化后的 (dr,dc) 当作图节点, 沿邻接边
#   叠加连线。原代码里 `out[total > 0] = templ_col` 等无条件赋值本身没有问题, 因为我们把连线
#   放到这些赋值之后, 以 O 色覆盖最终结果, 不会被后续步骤误擦除。
#
# d) 放置合法性
# 连线只落在已放置 stamp 的并集 bbox 内部: 水平边只覆盖两个相邻 bbox 组成的 1x(2w) 条带,
# 垂直边只覆盖 (2h)x1 条带。因此只要原始 stamps 全部在界内, 连线也自动在界内, 不需要新的
# 越界推理。Generator 继续保证 stamp 之间无像素重叠; 连线允许在共享 stamp 内交叉, 这是题意中的
# 第二步叠加效果, 不构成非法重叠。
#
# e) 视觉可观察性
# 解题者可先从输入找到模板和晶格, 再从输出看到两件事同时发生:
# - 所有晶格位置都出现模板副本, 原点副本为 O 色;
# - 相邻副本之间被 O 色直线串起来, 形成树/网结构。
# 由于样例强制同时出现横向与纵向连线, 2-3 组样例足以推出“复制后再连接”。
#
# f) 可视化整洁度
# 连线只用单像素厚度, 且只走模板 bbox 的中心十字, 不会在背景中随意拉出长走廊, 画面仍然规整。
# 颜色仍然只有 bg/T/O 三种, 输出不会因为额外装饰变得杂乱。
#
# g) 颜色变化可推断性
# 新增操作只使用已有的 O 色, 含义统一: 原点副本、重叠区、连接线都用 O 表示“特殊覆盖”。
# 没有基于颜色 id 的隐藏排序, 颜色变化完全由结构关系(原点/覆盖/邻接边)触发。


def _parse_input(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert not (not comps)

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
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(templ_col)]
    assert not (len(origin) != 1)
    orr, orc = origin[0]

    other_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert not (not other_cols)
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = int(math.gcd(*gcd_vals))
    gunit = max(1, int(gunit))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    return g, bg, H, W, templ_col, overlap_col, r0, c0, h, w, templ_rel, deltas


def _stamp_only(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    templ_rel: list[tuple[int, int]],
    deltas: list[tuple[int, int, int]],
) -> np.ndarray:
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


def _draw_connectors(
    out: np.ndarray,
    overlap_col: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    deltas: list[tuple[int, int, int]],
) -> None:
    H, W = out.shape
    nodes = {(int(dr), int(dc)) for dr, dc, _ in deltas}

    for dr, dc in sorted(nodes):
        if (int(dr), int(dc + 1)) in nodes:
            rr = int(r0 + dr * h + h // 2)
            cc0 = int(c0 + dc * w)
            cc1 = int(c0 + (dc + 1) * w + w - 1)
            if 0 <= rr < H:
                lo = max(0, min(cc0, cc1))
                hi = min(W - 1, max(cc0, cc1))
                out[int(rr), int(lo) : int(hi) + 1] = int(overlap_col)
        if (int(dr + 1), int(dc)) in nodes:
            cc = int(c0 + dc * w + w // 2)
            rr0 = int(r0 + dr * h)
            rr1 = int(r0 + (dr + 1) * h + h - 1)
            if 0 <= cc < W:
                lo = max(0, min(rr0, rr1))
                hi = min(H - 1, max(rr0, rr1))
                out[int(lo) : int(hi) + 1, int(cc)] = int(overlap_col)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a monochrome template from a singleton lattice, then connect neighboring copies.

    Input structure:
    - bg is the background. One large monochrome component is the template in color T.
    - Separate singleton pixels form a uniformly spaced 4-neighbor lattice; exactly one singleton is color T and marks the origin, while the rest are color O.

    Rule:
    - First recover each singleton's lattice offset from the T-colored origin and stamp a translated copy of the template at every offset, using bbox-scaled steps as in the base task.
    - The origin copy and any overlapped stamp pixels are recolored to O.
    - Then, for every pair of lattice cells that are orthogonally adjacent, draw an O-colored straight connector between the corresponding stamped copies along the copies' center row or center column.

    Key insight:
    - The singleton lattice still tells you where every copy goes, but the final output is not just the stamped set: the lattice edges themselves must be lifted into the stamp layout as horizontal and vertical O-colored links.
    """
    (
        _g,
        bg,
        H,
        W,
        templ_col,
        overlap_col,
        r0,
        c0,
        h,
        w,
        templ_rel,
        deltas,
    ) = _parse_input(grid)

    out = _stamp_only(H, W, bg, templ_col, overlap_col, r0, c0, h, w, templ_rel, deltas)
    _draw_connectors(out, overlap_col, r0, c0, h, w, deltas)
    return out


# --------------- generator (a1 + D12 variant) ---------------

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2

# Retry budgets
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


def _edge_orientations(cells: list[tuple[int, int]]) -> set[str]:
    cell_set = set(cells)
    dirs: set[str] = set()
    for r, c in cell_set:
        if (int(r), int(c + 1)) in cell_set:
            dirs.add("h")
        if (int(r + 1), int(c)) in cell_set:
            dirs.add("v")
    return dirs


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Lattice polyomino of *n* cells with strictly interior origin and both edge directions."""
    ext_r = random.randint(*LATTICE_EXTENT_RANGE)
    ext_c = random.randint(*LATTICE_EXTENT_RANGE)
    cells = _spanning_polyomino(n, ext_r, ext_c)
    if cells is None:
        return None
    if _edge_orientations(cells) != {"h", "v"}:
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
    T,
    O,
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
            grid[r0 + tr, c0 + tc] = T
        for dr, dc in deltas:
            grid[orr + dr * GUNIT, orc + dc * GUNIT] = T if (dr == 0 and dc == 0) else O

        try:
            out = main(grid)
        except Exception:
            continue

        # Reject accidental degeneracy: the D12 step must add at least one extra O pixel
        # beyond the base stamp-only result.
        (
            _g,
            _bg,
            H,
            W,
            templ_col,
            overlap_col,
            sr_base,
            sc_base,
            sh,
            sw,
            templ_rel,
            parsed_deltas,
        ) = _parse_input(grid)
        base = _stamp_only(H, W, _bg, templ_col, overlap_col, sr_base, sc_base, sh, sw, templ_rel, parsed_deltas)
        if int(np.sum(out != base)) <= 0:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, place stamps + singletons."""
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
        T,
        O,
        occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after %d attempts" % GENERATE_RETRIES)
