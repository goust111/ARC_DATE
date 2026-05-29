# Combo variant of a395ee82 (index 18)
# combo_spec: D5/1+D14/0
# Name: 象限旋转重叠消失
#
# design_analysis:
#   concretization:
#     - 输入仍是两块结构: 一块单色模板精灵(T色) + 一组均匀间距单例像素。
#     - 单例中恰有一个也是 T 色, 它给出晶格原点; 其余单例统一是 O 色。
#     - 输出不再沿用原题的“整 bbox 平移且永不相交”。为让 D5 的“重叠即消失”
#       真正可见, stamp 步长改成 `size-1`; 同时为让 D14 的 90 度旋转稳定成立,
#       模板 bbox 固定为 4x4 正方形并要求四个旋转互不相同。
#     - 具体小例子: 若某个单例位于原点左上方 `(-1,-1)`, 它对应的 stamp 会先旋转
#       90 度再放到 `(base_r-3, base_c-3)`; 若另一个单例在右上方 `(-1,+1)`, 它保持
#       0 度放到 `(base_r-3, base_c+3)`。因为步长只有 3, 它们都会与原点 stamp 在
#       角或边带上发生重叠; 这些重叠格不会变 O, 而是直接回到背景色。
#     - 与原题相比, 差异有两层协同: 先由象限决定 stamp 朝向, 再由这些不同朝向决定
#       哪些像素真的相撞并被抹掉; 所以“消失区域”本身依赖旋转。
#   complexity_calibration:
#     - 若仍用原题步长 4, 大多数样例不会重叠, D5 会退化成不可见改动。
#     - 若允许任意矩形模板, 90 度旋转会交换宽高并使放置逻辑复杂化, 还容易让样例
#       难以读懂。因此只保留一个小而稳定的自由度: 4x4 正方形模板 + 四个固定象限。
#     - generator 保证每个样例都出现四个对角象限, 并可选一个额外点, 让解题者从
#       少量样例中反复看到同一象限-旋转映射。
#   implicit_assumption_review:
#     - 原 solver 假设所有副本共用同一 `templ_rel`; 这里必须改为每个 delta 先按象限
#       旋转再计算覆盖。
#     - 原 solver 假设 `off_r = dr*h`, `off_c = dc*w` 会让副本刚好平铺不重叠; 这里
#       改成 `step = size-1` 才能触发可见干涉。
#     - 原 solver 将 `total >= 2` 统一染成 O; 这里必须删掉该着色并保留背景, 否则
#       无法体现“重叠消失”。
#     - D14 引入了一个新前提: 非原点单例不能落在坐标轴上, 否则象限归属不明确。
#       generator 显式只生成严格对角象限中的点, `main()` 也断言这一点。
#     - 融合后还新增一个更强前提: 旋转必须真的改变干涉结果。generator 会比较
#       “带旋转的消失输出”和“无旋转的消失输出”, 强制二者不同, 防止退化成纯 D5。
#   placement_legality:
#     - 所有 stamp 的锚点按 `step = 3` 计算, 放置窗口用所有 delta 的 min/max 求出,
#       保证每个旋转后的 4x4 盒子完整落在 22x22 内。
#     - 输入模板 bbox 与单例簇 bbox 仍保持至少 1 个背景格的分离。
#     - 单例像素禁止落在任一 stamp 曾覆盖过的格子上, 包括最终被抵消成背景的格子,
#       避免输入结构与输出足迹纠缠。
#   observability:
#     - 模板要求四个旋转都可区分, 因而解题者能从输出直接看到“右上 0 度、左上 90、
#       左下 180、右下 270”的规律。
#     - 同时又能看到某些相撞位置消失成背景, 而不是变 O; 且这些缺口并不总出现在
#       同一角, 因为它们受旋转后的形状控制。
#   cleanliness:
#     - 只用三种颜色 bg/T/O, 并将输出可见层次限制为: T 区、原点 O 区、背景缺口。
#     - 模板小、晶格点数少、额外点最多 1 个, 因此重叠存在但不会把整个画面搅乱。
#   color_inference:
#     - 颜色语义仍完全结构化: T 表示模板与普通可见副本, O 只表示原点副本独占可见
#       像素, 重叠则直接变回 bg。没有任何基于颜色 id 大小的隐藏规则。
#   post_edit_assumption_audit:
#     - `main()` 现在假设: 最大连通块是 4x4 方框内的模板; 恰有一个单例与模板同色;
#       其余单例共享颜色 O; 非原点单例都处于严格对角象限; 模板四旋转可区分。
#     - `generate_input()` 通过模板筛选、固定 lattice 家族、放置窗口检查、以及输出
#       对照验证, 分别保证了这些假设。

from __future__ import annotations

import math
import random
import sys
from collections import Counter

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_h, bbox_w, find_components, get_bg


GRID_SIZE = 22
BOX_SIZE = 4
STEP = BOX_SIZE - 1
TEMPLATE_AREA_RANGE = (7, 10)
SINGLETON_COUNT_RANGE = (5, 6)
GUNIT = 2

MIN_CANCEL_CELLS = 2
MIN_ORIGIN_VISIBLE = 2
MIN_TEMPLATE_VISIBLE = 6
MIN_ROTATION_DELTA = 3
MIN_CANCEL_DELTA = 1

SPANNING_RETRIES = 60
TEMPLATE_RETRIES = 140
SINGLETON_RETRIES = 100
GENERATE_RETRIES = 1200

BASE_DELTAS = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]
EXTRA_DELTAS = [
    (-2, -1),
    (-1, -2),
    (-2, 1),
    (-1, 2),
    (2, -1),
    (1, -2),
    (2, 1),
    (1, 2),
]
_GEN_CALL_INDEX = 0


def _rotate_cells(cells: list[tuple[int, int]], size: int, turns: int) -> list[tuple[int, int]]:
    """Rotate bbox-relative cells by 90-degree CCW steps inside a size x size box."""
    turns %= 4
    out = [(int(r), int(c)) for r, c in cells]
    for _ in range(turns):
        out = [(int(c), int(size - 1 - r)) for r, c in out]
    return out


def _quadrant_turns(dr: int, dc: int) -> int:
    """
    Map an origin-relative offset to the quadrant rotation.

    Upper-right -> 0, upper-left -> 1, lower-left -> 2, lower-right -> 3.
    The origin keeps the canonical orientation.
    """
    dr = int(dr)
    dc = int(dc)
    if dr == 0 and dc == 0:
        return 0
    assert dr != 0 and dc != 0
    if dr < 0 and dc > 0:
        return 0
    if dr < 0 and dc < 0:
        return 1
    if dr > 0 and dc < 0:
        return 2
    return 3


def _coverage_maps(
    shape: tuple[int, int],
    base_r: int,
    base_c: int,
    templ_cells: list[tuple[int, int]],
    size: int,
    deltas: list[tuple[int, int]],
    *,
    use_rotation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return total coverage count and coverage contributed by the origin copy."""
    H, W = map(int, shape)
    total = np.zeros((H, W), dtype=int)
    origin = np.zeros((H, W), dtype=int)
    for dr, dc in deltas:
        turns = _quadrant_turns(dr, dc) if use_rotation else 0
        stamp = _rotate_cells(templ_cells, size, turns)
        off_r = int(base_r + dr * STEP)
        off_c = int(base_c + dc * STEP)
        is_origin = int(dr) == 0 and int(dc) == 0
        for rr, cc in stamp:
            R = int(off_r + rr)
            C = int(off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    origin[R, C] += 1
    return total, origin


def _render_output(bg: int, templ_col: int, overlap_col: int, total: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Render the fused output from coverage maps."""
    out = np.full(total.shape, int(bg), dtype=int)
    out[(total == 1) & (origin == 0)] = int(templ_col)
    out[(total == 1) & (origin > 0)] = int(overlap_col)
    return out


def _render_variant(
    shape: tuple[int, int],
    bg: int,
    templ_col: int,
    overlap_col: int,
    base_r: int,
    base_c: int,
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
    *,
    use_rotation: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render an output and return `(out, total, origin)` for a chosen transform family."""
    total, origin = _coverage_maps(
        shape,
        base_r,
        base_c,
        templ_cells,
        BOX_SIZE,
        deltas,
        use_rotation=use_rotation,
    )
    return _render_output(bg, templ_col, overlap_col, total, origin), total, origin


def _variant_stats(
    bg: int,
    templ_col: int,
    overlap_col: int,
    base_r: int,
    base_c: int,
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
) -> dict[str, int]:
    fused_out, fused_total, fused_origin = _render_variant(
        (GRID_SIZE, GRID_SIZE),
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        templ_cells,
        deltas,
        use_rotation=True,
    )
    plain_out, plain_total, _ = _render_variant(
        (GRID_SIZE, GRID_SIZE),
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        templ_cells,
        deltas,
        use_rotation=False,
    )
    return {
        "cancel_cells": int(np.sum(fused_total >= 2)),
        "origin_visible": int(np.sum((fused_total == 1) & (fused_origin > 0))),
        "templ_visible": int(np.sum((fused_total == 1) & (fused_origin == 0))),
        "rotation_delta_cells": int(np.sum(fused_out != plain_out)),
        "cancel_delta_cells": int(np.sum((fused_total >= 2) != (plain_total >= 2))),
    }


def _touched_points(
    base_r: int,
    base_c: int,
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    """All cells visited by any rotated stamp before cancellation is applied."""
    pts: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        stamp = _rotate_cells(templ_cells, BOX_SIZE, _quadrant_turns(dr, dc))
        off_r = int(base_r + dr * STEP)
        off_c = int(base_c + dc * STEP)
        for rr, cc in stamp:
            pts.add((int(off_r + rr), int(off_c + cc)))
    return pts


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of n cells in a rows x cols box."""
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    cells = {start}
    frontier: list[tuple[int, int]] = []
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
    """Random 4-connected polyomino that touches all four sides of the bbox."""
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _is_rotation_distinct(cells: list[tuple[int, int]], size: int) -> bool:
    """Require four visually distinct quarter-turns inside the square bbox."""
    seen = set()
    for turns in range(4):
        seen.add(frozenset(_rotate_cells(cells, size, turns)))
    return len(seen) == 4


def _gen_template() -> list[tuple[int, int]] | None:
    """Square, 4-connected, spanning, rotationally asymmetric monochrome template."""
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(*TEMPLATE_AREA_RANGE)
        cells = _spanning_polyomino(area, BOX_SIZE, BOX_SIZE)
        if cells is None:
            continue
        if _is_rotation_distinct(cells, BOX_SIZE):
            return cells
    return None


def _gen_lattice(target_count: int) -> list[tuple[int, int]]:
    """All four diagonal quadrants are present; calls alternate between 5-point and 6-point coverage."""
    deltas = list(BASE_DELTAS)
    if int(target_count) == 6:
        deltas.append(random.choice(EXTRA_DELTAS))
    return deltas


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _place_singletons(
    deltas: list[tuple[int, int]],
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
    templ_cells: list[tuple[int, int]],
    base_r: int,
    base_c: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    expected: np.ndarray,
) -> np.ndarray | None:
    """Place the singleton lattice away from both the input template and all stamp footprints."""
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(base_r), int(base_r + BOX_SIZE - 1)
    tc0, tc1 = int(base_c), int(base_c + BOX_SIZE - 1)
    touched = _touched_points(base_r, base_c, templ_cells, deltas)

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = int(orr + min_dr * GUNIT)
        sr1 = int(orr + max_dr * GUNIT)
        sc0 = int(orc + min_dc * GUNIT)
        sc1 = int(orc + max_dc * GUNIT)
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        singleton_pixels = {(int(orr + dr * GUNIT), int(orc + dc * GUNIT)) for dr, dc in deltas}
        if singleton_pixels & touched:
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[int(base_r + tr), int(base_c + tc)] = int(templ_col)
        for dr, dc in deltas:
            rr = int(orr + dr * GUNIT)
            cc = int(orc + dc * GUNIT)
            grid[rr, cc] = int(templ_col if (dr, dc) == (0, 0) else overlap_col)

        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, expected):
            return grid

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp quadrant-rotated template copies from a singleton lattice, with overlap cells erased.

    Input structure:
    - The dominant color is the background. One large monochrome component is the template in color T.
    - A separated singleton lattice marks copy locations: exactly one singleton is also color T and serves as the lattice origin.
    - Every remaining singleton has color O, and all singleton offsets lie on one coarse evenly spaced grid.

    Rule:
    - Recover coarse lattice offsets from the unique T-colored singleton.
    - Copy the template once per offset, but move copies by `template_box_size - 1` cells per lattice step so nearby stamps can interfere.
    - Before placing each copy, rotate it by quadrant around the origin: upper-right 0 degrees, upper-left 90 degrees, lower-left 180 degrees, lower-right 270 degrees.
    - Pixels covered only by the origin copy become O, pixels covered only by non-origin copies stay T, and any pixel covered by two or more copies disappears to the background.

    Key insight: The singleton lattice tells you both where to stamp and which quarter-turn to use; because the stride is intentionally short, those rotated silhouettes collide, and the collision pattern itself is the evidence that overlap means erasure.
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
    assert h == w
    size = int(h)
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

    total = np.zeros((H, W), dtype=int)
    origin_cover = np.zeros((H, W), dtype=int)
    step = max(1, int(size) - 1)
    for dr, dc in deltas:
        turns = _quadrant_turns(dr, dc)
        stamp = _rotate_cells(templ_rel, size, turns)
        off_r = int(r0 + dr * step)
        off_c = int(c0 + dc * step)
        is_origin = int(dr) == 0 and int(dc) == 0
        for rr, cc in stamp:
            R = int(off_r + rr)
            C = int(off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    origin_cover[R, C] += 1

    return _render_output(bg, templ_col, overlap_col, total, origin_cover)


def _gen_attempt(target_count: int) -> np.ndarray | None:
    """Build the fused output first, then back out a matching input."""
    bg, templ_col, overlap_col = random.sample(range(10), 3)

    templ_cells = _gen_template()
    if templ_cells is None:
        return None

    deltas = _gen_lattice(target_count)
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    base_r_lo = -min_dr * STEP
    base_r_hi = GRID_SIZE - max_dr * STEP - BOX_SIZE
    base_c_lo = -min_dc * STEP
    base_c_hi = GRID_SIZE - max_dc * STEP - BOX_SIZE
    if base_r_lo > base_r_hi or base_c_lo > base_c_hi:
        return None

    base_r = random.randint(base_r_lo, base_r_hi)
    base_c = random.randint(base_c_lo, base_c_hi)

    stats = _variant_stats(bg, templ_col, overlap_col, base_r, base_c, templ_cells, deltas)
    if stats["cancel_cells"] < MIN_CANCEL_CELLS:
        return None
    if stats["origin_visible"] < MIN_ORIGIN_VISIBLE:
        return None
    if stats["templ_visible"] < MIN_TEMPLATE_VISIBLE:
        return None
    if stats["rotation_delta_cells"] < MIN_ROTATION_DELTA:
        return None
    if stats["cancel_delta_cells"] < MIN_CANCEL_DELTA:
        return None

    expected, _total, _origin = _render_variant(
        (GRID_SIZE, GRID_SIZE),
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        templ_cells,
        deltas,
        use_rotation=True,
    )
    return _place_singletons(
        deltas,
        min_dr,
        max_dr,
        min_dc,
        max_dc,
        templ_cells,
        base_r,
        base_c,
        bg,
        templ_col,
        overlap_col,
        expected,
    )


def generate_input() -> np.ndarray:
    global _GEN_CALL_INDEX
    target_count = SINGLETON_COUNT_RANGE[_GEN_CALL_INDEX % len(SINGLETON_COUNT_RANGE)]
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(target_count)
        if grid is not None:
            _GEN_CALL_INDEX += 1
            return grid
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
