# Combo variant of a395ee82 (index 15)
# combo_spec: D4/4+D9/0
# Name: 擦除保留标记

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis
# combo_variant:
#   concretization_and_behavior_trace:
#   - 保留原题输入骨架：一个大号单色模板 T，加上一组等间距单例点；其中唯一的 T 色单例仍是晶格原点，其余单例共享 O 色。
#   - 小例子：若晶格 delta 为 {(-1,0), (0,0), (0,1), (1,0), (1,1)}，原题会画出 5 份模板，其中原点那一份整块改成 O。
#   - D4/4 先改成“原点副本不落笔”：仍计算该副本位置，但该位置保持背景色，得到一个模板形状的缺口。
#   - D9/0 再改成“单例最后覆写回来”：无论单例落在背景还是某份 stamp 上，输出最后都恢复这些输入标记。
#   - 融合后的可见效果是：输出中既有一个缺失的原点模板槽位，也仍保留 5-6 个源标记；若某个 O 标记恰好压在非原点 T-stamp 上，就会变成 stamp 内的异色点。
#   complexity_calibration:
#   - 仍只使用 bg/T/O 三色，避免把“擦除”和“保留标记”混成额外的颜色映射题。
#   - 继续保留唯一 T 原点、其余单例同色、固定 GUNIT=2、origin 严格位于晶格内部，保证缺口位置可推断。
#   - 保持 stamp 之间无像素重叠，使 D4 的“缺口”干净可见；D9 的可观察性则通过让至少一个保留标记压在未来 stamp 上来保证。
#   hidden_assumption_audit:
#   - `template = max(comps, key=area)` 仍成立，因为输入里除模板外都还是 area==1 的独立单例。
#   - `origin` 仍由“唯一一个颜色等于模板色的单例”识别；融合不改变这条输入约束。
#   - `gcd(offsets)` 推回晶格单位仍成立，因为单例晶格结构未改。
#   - 原题依赖“先整体画 T，再把 origin / overlap 改成 O”；融合后该前提失效，因为 origin 副本应完全不绘制，且单例还要在最终层覆盖回去。
#   - 因此实现上必须拆成两层：先渲染 erase-stamp 层，再可选叠加 singleton 层。
#   placement_legality:
#   - 继续要求所有 stamp 完全位于 22x22 内，且 stamp 彼此无重叠。
#   - 输入里模板组件与单例晶格 bbox 仍保持至少 1 格背景间隔，防止组件识别时并到一起。
#   - 与原题不同，generator 不再禁止单例落在“未来某个非原点 stamp”的像素上；这是保留标记可见性的关键。
#   visual_observability:
#   - 解题者能在 2-4 组样例中同时看到两件事：原模板位置变成模板形缺口；单例点没有被删除，而且有些点压在复制出的模板内部。
#   - 这些现象完全由空间结构给出，不依赖颜色编号大小、扫描顺序等不可视因素。
#   cleanliness_and_inferability:
#   - 不引入 stamp-stamp 重叠，避免同时混入“副本冲突着色”和“保留标记覆写”两种干扰来源。
#   - 单例只以 T/O 两色出现，并且都保持输入原位，因而“最后覆写回去”的规则能从位置对应中归纳出来。
#   new_main_assumptions_generator_must_satisfy:
#   - 必须恰有一个 T 色单例，且至少还有一个非原点单例。
#   - 必须存在至少一个非零 delta，让 gcd 能推出晶格单位。
#   - 至少一个保留单例要压在 erase-stamp 层的非背景像素上，否则 D9 在样例里会退化成“只是把点留在空白处”。
#   - generator 用 `stamp_hits >= 1` 和 `visible_keep_diff >= 1` 显式保证上述新前提。


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
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
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert other_cols
    marker_col = int(other_cols[0])
    assert all(int(col) == int(marker_col) for col in other_cols)

    deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = int(math.gcd(*gcd_vals))
    gunit = max(1, int(gunit))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    return {
        "bg": int(bg),
        "H": int(H),
        "W": int(W),
        "templ_col": int(templ_col),
        "marker_col": int(marker_col),
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
    r0 = int(scene["r0"])
    c0 = int(scene["c0"])
    h = int(scene["h"])
    w = int(scene["w"])
    templ_rel = list(scene["templ_rel"])
    deltas = list(scene["deltas"])
    singletons = list(scene["singletons"])

    non_origin_cover = np.zeros((H, W), dtype=int)
    origin_cover = np.zeros((H, W), dtype=int)

    for dr, dc, col in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        is_origin = int(col) == int(templ_col)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                if is_origin:
                    origin_cover[R, C] += 1
                else:
                    non_origin_cover[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[non_origin_cover > 0] = int(templ_col)

    if keep_singletons:
        for rr, cc, col in singletons:
            out[int(rr), int(cc)] = int(col)

    return out


def _solve(grid: np.ndarray, *, keep_singletons: bool) -> np.ndarray:
    scene = _extract_scene(grid)
    return _render_scene(scene, keep_singletons=keep_singletons)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a monochrome template from a singleton lattice, but erase the origin copy while preserving the source dots.

    Input structure:
    - The grid has one large monochrome template in color T on a uniform background.
    - Elsewhere there is a lattice of isolated singletons: exactly one singleton is also T and marks the lattice origin, while all other singletons share color O.
    - The singleton lattice is evenly spaced and remains a separate object group from the template in the input.

    Rule:
    - Measure every singleton offset from the unique T-colored origin, normalize those offsets by the lattice spacing, and scale them by the template bbox height and width.
    - Stamp the template at every non-origin lattice position using color T, but do not draw the origin-position copy at all, so that slot stays background unless another later layer restores something there.
    - After the erase-style stamp layer is complete, write all original singleton pixels back at their exact input coordinates and original colors, letting those retained markers override any stamp beneath them.

    Key insight: The T singleton still identifies which translated copy corresponds to the template itself, but in this fusion it marks the missing copy, and the input lattice dots survive as a final overlay instead of being consumed by stamping.
    """
    return _solve(grid, keep_singletons=True)


# --------------- generator (combo D4 erase + D9 keep_source) ---------------

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2

# Combo observability constraints
MIN_STAMPED_SINGLETONS = 1
MIN_VISIBLE_KEEP_DIFF = 1

# Retry budgets
SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
SINGLETON_RETRIES = 160
GENERATE_RETRIES = 600
_GENERATE_CALL_COUNT = 0


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


def _variant_metrics(grid: np.ndarray) -> tuple[int, int, int, int]:
    """
    Measure whether the combo behavior is visibly active.

    Returns:
    - stamp_hits: kept singletons that sit on a non-background pixel of the erase-stamp layer
    - visible_keep_diff: kept singletons whose color differs from the erase-stamp layer beneath
    - hole_pixels: background pixels inside the origin template footprint in erase-only output
    - origin_template_area: number of pixels in the template shape
    """
    scene = _extract_scene(grid)
    erase_only = _render_scene(scene, keep_singletons=False)
    bg = int(scene["bg"])
    templ_col = int(scene["templ_col"])
    r0 = int(scene["r0"])
    c0 = int(scene["c0"])
    templ_rel = list(scene["templ_rel"])

    stamp_hits = 0
    visible_keep_diff = 0
    for rr, cc, col in scene["singletons"]:
        base_col = int(erase_only[int(rr), int(cc)])
        if base_col != int(bg):
            stamp_hits += 1
        if base_col != int(col):
            visible_keep_diff += 1

    hole_pixels = 0
    for rr, cc in templ_rel:
        if int(erase_only[r0 + rr, c0 + cc]) == int(bg):
            hole_pixels += 1

    return int(stamp_hits), int(visible_keep_diff), int(hole_pixels), int(len(templ_rel))


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
    marker_col: int,
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
            grid[r0 + tr, c0 + tc] = templ_col
        for dr, dc in deltas:
            rr = orr + dr * GUNIT
            cc = orc + dc * GUNIT
            grid[rr, cc] = templ_col if (dr == 0 and dc == 0) else marker_col

        try:
            combo_out = main(grid)
            stamp_hits, visible_keep_diff, hole_pixels, origin_area = _variant_metrics(grid)
        except Exception:
            continue

        if stamp_hits < MIN_STAMPED_SINGLETONS:
            continue
        if visible_keep_diff < MIN_VISIBLE_KEEP_DIFF:
            continue
        if hole_pixels != origin_area:
            continue
        if not np.array_equal(combo_out, _solve(grid, keep_singletons=True)):
            continue

        return grid

    return None


def _gen_attempt(n_sing: int | None = None) -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, place template + singletons."""
    bg, templ_col, marker_col = random.sample(range(10), 3)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None

    if n_sing is None:
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

    occupied: set[tuple[int, int]] = set()
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
        marker_col,
    )


def generate_input() -> np.ndarray:
    global _GENERATE_CALL_COUNT

    lo, hi = SINGLETON_COUNT_RANGE
    branch_values = list(range(lo, hi + 1))
    preferred = branch_values[_GENERATE_CALL_COUNT % len(branch_values)]
    _GENERATE_CALL_COUNT += 1
    ordered_counts = [preferred] + [n for n in branch_values if n != preferred]

    per_branch_retries = max(1, GENERATE_RETRIES // len(ordered_counts))
    for n_sing in ordered_counts:
        for _ in range(per_branch_retries):
            result = _gen_attempt(n_sing=n_sing)
            if result is not None:
                return result

    for _ in range(max(0, GENERATE_RETRIES - per_branch_retries * len(ordered_counts))):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after %d attempts" % GENERATE_RETRIES)
