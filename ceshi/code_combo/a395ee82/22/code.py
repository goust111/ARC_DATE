# Combo variant of a395ee82 (index 22)
# combo_spec: D9/0+D10/0
# Name: 保留标记紧凑裁剪

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 原题先依据单例晶格重建所有 stamp，再返回整张 22x22 画布；所有输入单例都会消失。
# - 本融合版仍先按原题规则生成 stamp 层：原点对应副本改成 O，若发生多重覆盖则也改成 O。
# - 然后把输入里的 5-6 个单例原位覆写回去，颜色保持输入值不变，因此它们既可能留在背景上，也可能压在 stamp 内部形成小异色点。
# - 最后不再返回整张 22x22，而是对“stamp 层 + 保留单例”这个并集做最小外接矩形裁剪。若某个保留单例落在所有 stamp 之外，它会把裁剪边界向外推大。
# - 小例子：若旧题 stamp 结果占 rows 4..12, cols 6..14，而保留单例里有一个 O 点在 (3, 10)、一个 T 点落在某个 O-stamp 内部，则旧题输出仍是 22x22；
#   本题输出会裁到 rows 3..12, cols 6..14，同时内部那个 T 点会覆盖掉原本的 O。
# b) 复杂度校准
# - 仍只使用 bg/T/O 三色，不引入额外颜色自由度，避免把规律变成人类难以归纳的颜色映射题。
# - 保持模板单色、单例晶格 spacing、bbox 步长映射不变，只在“输出合成”和“最终裁剪对象集合”两处改动。
# - generator 强制两类可见证据同时出现：至少一个保留单例覆盖某个 stamp 像素并改变其颜色；至少一个保留单例位于 stamp 外部，从而扩张 tight crop。
# c) 隐式假设审查
# - `template = max(comps, key=area)` 仍要求模板是输入里唯一最大的连通块；generator 保持模板与单例分离，因此成立。
# - `origin` 仍依赖“恰有一个单例颜色等于模板色 T”；融合版不改输入识别方式，这个前提不变。
# - `gunit = gcd(nonzero deltas)` 仍要求晶格至少有一个非零偏移；5-6 个严格内点晶格继续保证这一点。
# - 原题把 stamp 层直接作为最终输出；这里该假设失效，因为还要执行“覆写回原单例”与“按并集裁剪”。
# - 新增假设：tight crop 必须针对最终保留单例后的前景并集，而不是仅针对 stamp 层；generator 会显式检查 kept bbox 与 stamp bbox 的差异。
# d) 放置合法性
# - 输入中模板 bbox 与单例 cluster bbox 仍保持至少 1 格背景间隔，避免 solver 在解析输入时把它们误并。
# - 所有 stamp 仍完全落在 22x22 内，且继续禁止 stamp 彼此像素重叠，避免把本融合任务和重叠解析纠缠在一起。
# - 单例允许落到未来 stamp 上，因为这是“保留标记覆盖 stamp”所需的关键行为；但 final kept bbox 仍必须小于整张 22x22，确保 tight crop 可见。
# e) 视觉可观察性
# - 解题者会同时看到三件事：模板复制结构、输出里仍残留的晶格点、以及输出尺寸随“复制结构 + 残留点”共同变化。
# - 至少一个残留点压在 stamp 内部，至少一个残留点把边界往外推，因此 2-4 组样例足以归纳“点会保留，而且裁剪要把这些点也算进去”。
# f) 可视化整洁度
# - 只保留单层 stamp 与少量单例点，元素类型仍然很少；继续禁止 stamp-over-stamp overlap，避免输出混成大片难分的 O 区。
# - crop 反而去掉大量无关背景，使“保留标记”与“边界被扩张”更容易被肉眼看到。
# g) 变化的可推断性
# - 颜色行为依然只有 T 与 O 两种前景角色：T 是模板色与原点单例色，O 是非原点单例色以及原点 stamp 色。
# - 保留单例的颜色不做新映射，只通过位置与覆盖关系改变最终像素；因此人类可从结构线索而非颜色编号本身推断规则。


def _foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


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
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    return {
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a stamped template pattern from a singleton lattice, keep the lattice dots, and tightly crop their union.

    Input structure:
    - One large monochrome template component uses color T on a uniform background bg.
    - Separate singleton dots form a coarse lattice; exactly one dot is also T and marks the lattice origin, while the others share color O.
    - The singleton cluster is isolated from the template in the input, but some of those dots may coincide with future stamped pixels after reconstruction.

    Rule:
    - Measure every singleton offset relative to the unique T-colored origin, normalize those offsets by the lattice spacing, and scale them by the template bbox height and width.
    - Stamp the template at every recovered offset; recolor the origin's own stamped copy to O, and recolor any multiply covered stamp pixels to O as well.
    - After the stamp layer is complete, restore every original singleton at its input coordinate with its original color, so retained dots override any stamp underneath them.
    - Finally crop the result to the smallest rectangle containing every non-background pixel from both the stamps and the restored singleton markers.

    Key insight:
    - The singleton lattice still encodes exactly the same translation pattern as the source task, but the answer is no longer just the stamp layer: the source dots survive and can both puncture stamped copies and enlarge the tight crop boundary.
    """
    scene = _extract_scene(grid)
    out = _render_scene(scene, keep_singletons=True)
    bg = int(scene["bg"])
    r0, c0, r1, c1 = _foreground_bbox(out != int(bg))
    return out[r0 : r1 + 1, c0 : c1 + 1]


# --------------- generator ---------------

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2
OUTPUT_MARGIN = 2

MIN_STAMP_HITS = 1
MIN_VISIBLE_PUNCTURES = 1
MIN_CROP_AREA_GAIN = 1

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
SINGLETON_RETRIES = 180
GENERATE_RETRIES = 700


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


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    scene = _extract_scene(grid)
    bg = int(scene["bg"])
    stamp_only = _render_scene(scene, keep_singletons=False)
    kept_full = _render_scene(scene, keep_singletons=True)

    stamp_bbox = _foreground_bbox(stamp_only != int(bg))
    kept_bbox = _foreground_bbox(kept_full != int(bg))
    stamp_shape = (stamp_bbox[2] - stamp_bbox[0] + 1, stamp_bbox[3] - stamp_bbox[1] + 1)
    kept_shape = (kept_bbox[2] - kept_bbox[0] + 1, kept_bbox[3] - kept_bbox[1] + 1)

    stamp_hits = 0
    visible_punctures = 0
    visible_overrides = 0
    for rr, cc, col in scene["singletons"]:
        if int(stamp_only[int(rr), int(cc)]) != int(bg):
            stamp_hits += 1
            if int(stamp_only[int(rr), int(cc)]) != int(col):
                visible_punctures += 1
        if int(stamp_only[int(rr), int(cc)]) != int(col):
            visible_overrides += 1

    return {
        "stamp_hits": int(stamp_hits),
        "visible_punctures": int(visible_punctures),
        "visible_overrides": int(visible_overrides),
        "stamp_bbox": stamp_bbox,
        "kept_bbox": kept_bbox,
        "stamp_shape": tuple(map(int, stamp_shape)),
        "kept_shape": tuple(map(int, kept_shape)),
        "crop_area_gain": int(_bbox_area(kept_bbox) - _bbox_area(stamp_bbox)),
        "expanded_sides": int(sum(int(a != b) for a, b in zip(stamp_bbox, kept_bbox))),
    }


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
    orr_lo = -min_dr * GUNIT
    orr_hi = GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo = -min_dc * GUNIT
    orc_hi = GRID_SIZE - 1 - max_dc * GUNIT
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
            out = main(grid)
            metrics = _behavior_metrics(grid)
        except Exception:
            continue

        kept_shape = tuple(metrics["kept_shape"])
        if metrics["stamp_hits"] < MIN_STAMP_HITS:
            continue
        if metrics["visible_punctures"] < MIN_VISIBLE_PUNCTURES:
            continue
        if metrics["crop_area_gain"] < MIN_CROP_AREA_GAIN:
            continue
        if kept_shape[0] >= GRID_SIZE or kept_shape[1] >= GRID_SIZE:
            continue
        if out.shape != kept_shape:
            continue

        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
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

    r0_lo = OUTPUT_MARGIN - min_dr * h
    r0_hi = GRID_SIZE - OUTPUT_MARGIN - max_dr * h - h
    c0_lo = OUTPUT_MARGIN - min_dc * w
    c0_hi = GRID_SIZE - OUTPUT_MARGIN - max_dc * w - w
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
        T,
        O,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
