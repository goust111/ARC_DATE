# Variant of a395ee82
# Dimension: D9 — 输入中单例像素的保留/移除 (trajectory)
# Change: 单例像素在输出中被移除(消失)，模板被原点stamp覆盖(重着色为O)... → 保留单例像素

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis
# ---
# variant:
#   core_change:
#   - 原题输出只保留 stamp 结果；晶格中的单例像素全部消失。
#   - 本变体改成“先按原题规则画出所有 stamp，再把输入中的单例像素原位覆写回去”。
#   - 因此输出里会同时出现：原模板平移复制出的大片单色块，以及仍然留在原处的 5-6 个单例点。
#   - 当某个单例恰好落在未来某个 stamp 的像素上时，输出会在 stamp 内部留下一个优先级更高的异色点。
#   concretization_and_trace:
#   - 例：模板色为 T，非原点单例色为 O。旧题里原点 delta=(0,0) 对应的模板副本整块改成 O；其他副本保持 T。
#   - 若某个输入单例 O 恰好位于某个未来 T-stamp 的像素坐标上，旧题输出该位置会是 T；新题会在最后一步把该位置覆写回 O。
#   - 若输入原点单例 T 位于某个未来 O-stamp 或背景位置上，旧题该处分别会是 O 或 bg；新题则改回 T。
#   - 变化发生在“输出合成”环节，不改变模板识别、原点识别、晶格间距求法、delta 到 stamp 偏移的映射。
#   complexity_calibration:
#   - 不引入新的颜色类别，仍然只用 bg/T/O 三色，避免把 D9 变化和额外颜色映射混在一起。
#   - 保持模板单色、晶格规则、间距规则不变，防止任务退化成另一类匹配题。
#   - 为了让 D9 肉眼可见，generator 不再禁止“单例落在未来 stamp 像素上”，反而要求至少出现一个这样的点。
#   hidden_assumption_audit:
#   - `template = max(comps, key=area)` 依赖模板始终是最大连通块；变体不改变输入，所以仍成立。
#   - `origin` 通过“唯一一个颜色等于模板色的单例”识别；变体只改输出，输入里这条仍成立。
#   - 原题 `out[(total == 1) & (center > 0)] = overlap_col` 默认原点 stamp 可以无条件覆盖其所有像素；变体下这条只适用于 stamp 层本身，
#     不能再视为最终结果，因为单例像素需要在最后覆写回来。
#   - 原题最终 `return out` 隐含“输入单例不保留”；这是 D9 中唯一必须推翻的旧前提。
#   placement_legality:
#   - 输入里模板 bbox 与单例晶格 bbox 仍保持至少 1 格背景间隔，避免 solver 在输入阶段把它们误并到一起。
#   - 所有 stamp 仍完全位于 22x22 内，且继续要求彼此无像素重叠，避免把 D9 与重叠逻辑耦合。
#   - 新增合法性要求：至少一个单例坐标必须落在旧题输出的非背景像素上，且至少一个单例在旧题输出里的颜色要与自身不同，
#     这样“保留单例”既参与了 stamp 覆盖关系，也能在输入输出样例里产生可见差异。
#   visual_observability:
#   - 人类会看到：大片模板复制图案已经生成，但右下/左侧仍残留原始晶格点；其中一部分点嵌在模板内部，成为孤立异色点。
#   - 由于所有样例都复用同一机制“单例最后覆写”，2-4 组样例足以归纳，不依赖颜色 id 排序或隐藏分支。
#   visual_cleanliness_and_color_inference:
#   - 仍使用三色，单例颜色只可能是 T 或 O，其中 O 已在旧题里承担“原点 stamp / 重叠”颜色，因此异色点的来源可由位置推断。
#   - 保持 stamp 无重叠，避免输出中同时混入“多重 stamp 冲突”和“保留单例冲突”两种叠加来源。
#   new_main_assumptions_generator_must_satisfy:
#   - 单例仍是 area==1 的独立分量。
#   - 恰有一个单例颜色等于模板色。
#   - 至少存在一个非零 delta，使 gcd 推导出晶格单位。
#   - generator 通过严格内点 origin、晶格 5-6 个点、GUNIT=2、模板与晶格分离来保证这些前提。


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
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

    return {
        "grid": g,
        "bg": int(bg),
        "H": int(H),
        "W": int(W),
        "templ_col": int(templ_col),
        "overlap_col": int(overlap_col),
        "r0": int(r0),
        "c0": int(c0),
        "h": int(h),
        "w": int(w),
        "templ_rel": templ_rel,
        "singletons": singletons,
        "deltas": deltas,
    }


def _render_scene(scene: dict[str, object], *, keep_singletons: bool) -> np.ndarray:
    bg = int(scene["bg"])
    H = int(scene["H"])
    W = int(scene["W"])
    templ_col = int(scene["templ_col"])
    overlap_col = int(scene["overlap_col"])
    r0 = int(scene["r0"])
    c0 = int(scene["c0"])
    h = int(scene["h"])
    w = int(scene["w"])
    templ_rel = list(scene["templ_rel"])
    deltas = list(scene["deltas"])
    singletons = list(scene["singletons"])

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

    if keep_singletons:
        for rr, cc, col in singletons:
            out[int(rr), int(cc)] = int(col)

    return out


def _solve(grid: np.ndarray, *, keep_singletons: bool) -> np.ndarray:
    scene = _extract_scene(grid)
    return _render_scene(scene, keep_singletons=keep_singletons)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a monochrome template from a singleton lattice, while keeping the source dots visible.

    Input structure:
    - The grid has one large monochrome template in color T.
    - Elsewhere, isolated lattice dots mark translation offsets: exactly one source dot is also T (the origin), while the remaining dots share color O.
    - The singleton dots lie on a uniform coarse lattice and stay separate from the template in the input.

    Rule:
    - Read every singleton offset relative to the unique T-colored origin, normalize by the lattice spacing, and scale offsets by the template bbox height/width.
    - Stamp the template at every derived offset; the origin's own copy is recolored to O, and any multi-covered stamp pixels are also recolored to O.
    - After all stamps are drawn, restore the original singleton dots at their exact input coordinates and colors, so those dots override any stamp pixel beneath them.

    Key insight: The singleton lattice still tells you where all template copies go, but unlike the base task the source dots are not erased at the end, so retained dots can appear as small color anomalies inside stamped copies.
    """
    return _solve(grid, keep_singletons=True)


# --------------- generator (a1 + D9 keep_source) ---------------

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2

# D9 observability constraints
MIN_STAMPED_SINGLETONS = 1
MIN_VISIBLE_KEEP_DIFF = 1

# Retry budgets
SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
SINGLETON_RETRIES = 120
GENERATE_RETRIES = 500


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
    """Return (min_dr, max_dr, min_dc, max_dc) for a list of (dr, dc) offsets."""
    min_dr = min(dr for dr, dc in deltas)
    max_dr = max(dr for dr, dc in deltas)
    min_dc = min(dc for dr, dc in deltas)
    max_dc = max(dc for dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _variant_metrics(grid: np.ndarray) -> tuple[int, int]:
    """Measure whether kept singletons visibly interact with future stamps."""
    scene = _extract_scene(grid)
    old_out = _render_scene(scene, keep_singletons=False)
    bg = int(scene["bg"])
    stamp_hits = 0
    visible_keep_diff = 0
    for rr, cc, col in scene["singletons"]:
        if int(old_out[int(rr), int(cc)]) != int(bg):
            stamp_hits += 1
        if int(old_out[int(rr), int(cc)]) != int(col):
            visible_keep_diff += 1
    return int(stamp_hits), int(visible_keep_diff)


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

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[r0 + tr, c0 + tc] = T
        for dr, dc in deltas:
            rr = orr + dr * GUNIT
            cc = orc + dc * GUNIT
            grid[rr, cc] = T if (dr == 0 and dc == 0) else O

        try:
            kept_out = main(grid)
            stamp_hits, visible_keep_diff = _variant_metrics(grid)
        except Exception:
            continue

        if stamp_hits < MIN_STAMPED_SINGLETONS or visible_keep_diff < MIN_VISIBLE_KEEP_DIFF:
            continue

        if not np.array_equal(kept_out, _solve(grid, keep_singletons=True)):
            continue

        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, place template + singletons."""
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
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after %d attempts" % GENERATE_RETRIES)
