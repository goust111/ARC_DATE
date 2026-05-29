# Combo variant of a395ee82 (index 21)
# combo_spec: D6/1+D14/0
# Name: 各向异性间距象限旋转

from __future__ import annotations

import math
import os
import random
import sys
from collections import Counter

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
SOLVER_DIR = os.path.join(ROOT_DIR, "solvers")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the combo variant
# -------------------------------------
# design_analysis:
#   concretization:
#     - 输入仍由两部分组成：一个大号单色模板 T，以及一簇与之分离的单例像素。
#     - 这簇单例里恰有一个也是 T 色，它继续充当原点；其它单例全是 O 色。
#     - 融合后的变化分两层同时发生：
#       1) D6/1: 单例晶格在行、列方向用不同像素间距采样。这里固定为 ROW_GUNIT=3、COL_GUNIT=2，
#          所以输入里“往上一格”和“往左一格”看起来并不等长。
#       2) D14/0: 还原出晶格单位后，输出中的每个 stamp 不再都保持同朝向，而是由其相对原点的象限决定旋转角度。
#     - 小例子：若单例相对原点的像素偏移为 (-3,+2), (-3,-2), (+3,-2), (+3,+2)，它们应解读为
#       晶格位移 (-1,+1), (-1,-1), (+1,-1), (+1,+1)。输出里四个位置都以 4x4 模板盒为步长平移，
#       但右上保持 0°，左上转 90°，左下转 180°，右下转 270°。
#     - 和原题相比，原题会把所有非零偏移用一个共同 gcd 解释，并把所有 stamp 画成同一朝向；融合后这两处都必须改。
#   complexity_calibration:
#     - 只保留两个新增自由度：一对固定的不等晶格单位，以及四象限到四个 quarter-turn 的固定映射。
#     - 模板、颜色语义、单例角色仍保持简洁，避免再引入额外条件分支。
#     - 生成器始终覆盖四个严格对角象限，并可额外重复一个象限，保证少量样例就能重复看到旋转规律。
#   implicit_assumption_review:
#     - 原 solver 的单一 `gunit` 假设失效，必须拆成 `row_unit` 和 `col_unit` 分别由各轴 gcd 恢复。
#     - 原 solver 默认每个 stamp 共享同一个 `templ_rel`；现在必须先按象限旋转，再执行平移。
#     - 90° 旋转若搭配非方形 bbox，会出现宽高互换并破坏固定锚点步长；因此这里显式改用 4x4 方形模板并要求四向旋转互异。
#     - 轴向点 `(dr==0 xor dc==0)` 会落在象限边界，旋转规则不再唯一；生成器只产出原点和严格对角点，`main()` 也断言这一点。
#   placement_legality:
#     - 输出阶段所有 stamp 都占据互不重叠的 4x4 盒子，锚点按 `BOX_SIZE` 的整数倍移动，因此旋转不会导致越界或盒间碰撞。
#     - 输入阶段单例坐标用 `(ROW_GUNIT, COL_GUNIT)` 计算，并保持与模板 bbox 至少隔 1 个背景格。
#     - 另外还避开整个预期输出会占据的像素区域，确保输入画面里模板与单例簇不会压到将来 stamp 的位置上，视觉上更干净。
#   visual_observability:
#     - 人眼可以直接看出单例簇在竖直方向步长为 3、水平步长为 2，因此“两个轴分别解码”是可见的。
#     - 输出中同一个模板会在四个象限出现四种朝向，而且模板被约束为四向旋转互不相同，因此“象限控制旋转”也能从结果反推出。
#     - 这两个线索协同出现：输入负责揭示各向异性采样，输出负责揭示象限旋转。
#   cleanliness:
#     - 仍只使用三种颜色和两类对象；输入保持“一块模板 + 一簇单例”的原始简洁布局。
#     - 输出禁止 stamp 像素重叠，避免 O 色同时承担“原点副本”和“重叠副本”时造成局部难以判读。
#   color_inference:
#     - 颜色语义不变：T 是模板本色，O 是原点副本色，同时也保留为潜在重叠色。
#     - 旋转与 spacing 的判定完全依赖几何关系，不依赖颜色编号大小或任何人类不可见信息。
#   post_edit_assumptions:
#     - `main()` 新增假设包括：模板 bbox 为方形；单例在两个轴上都至少出现一个非零偏移；
#       非原点单例都位于严格对角象限；行列晶格单位确实不相等。`generate_input()` 逐项构造满足这些条件的样本。

GRID_SIZE = 22
BOX_SIZE = 4
TEMPLATE_AREA_RANGE = (7, 10)
SINGLETON_COUNT_RANGE = (5, 6)
ROW_GUNIT = 3
COL_GUNIT = 2

SPANNING_RETRIES = 60
TEMPLATE_RETRIES = 120
SINGLETON_RETRIES = 120
GENERATE_RETRIES = 500

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


def _axis_unit(offsets: list[int]) -> int:
    vals = [abs(int(x)) for x in offsets if int(x) != 0]
    assert vals
    return max(1, int(math.gcd(*vals)))


def _rotate_cells(cells: list[tuple[int, int]], size: int, turns: int) -> list[tuple[int, int]]:
    """Rotate bbox-relative cells by 90-degree CCW steps inside a size x size box."""
    turns %= 4
    out = [(int(r), int(c)) for r, c in cells]
    for _ in range(turns):
        out = [(int(c), int(size - 1 - r)) for r, c in out]
    return out


def _quadrant_turns(dr: int, dc: int) -> int:
    """
    Map an origin-relative lattice offset to the quadrant rotation.

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


def _decode_scene(grid: np.ndarray) -> dict:
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
    templ_rel = [(int(r) - r0, int(c) - c0) for r, c in templ_coords]

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

    return {
        "grid": g,
        "bg": bg,
        "H": H,
        "W": W,
        "templ_col": templ_col,
        "overlap_col": overlap_col,
        "base_r": r0,
        "base_c": c0,
        "size": h,
        "templ_rel": templ_rel,
        "singletons": singletons,
        "origin_r": int(orr),
        "origin_c": int(orc),
    }


def _normalized_deltas(scene: dict) -> list[tuple[int, int, int]]:
    raw = [
        (int(r - scene["origin_r"]), int(c - scene["origin_c"]), int(col))
        for (r, c, col) in scene["singletons"]
    ]
    row_unit = _axis_unit([dr for dr, _dc, _col in raw])
    col_unit = _axis_unit([dc for _dr, dc, _col in raw])
    assert row_unit != col_unit
    deltas = [(int(dr // row_unit), int(dc // col_unit), int(col)) for dr, dc, col in raw]
    for dr, dc, _col in deltas:
        if dr == 0 and dc == 0:
            continue
        assert dr != 0 and dc != 0
    return deltas


def _render_output(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    base_r: int,
    base_c: int,
    size: int,
    templ_cells: list[tuple[int, int]],
    deltas: list[tuple[int, int]],
    *,
    rotate_by_quadrant: bool,
) -> np.ndarray:
    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)
    for dr, dc in deltas:
        turns = _quadrant_turns(dr, dc) if rotate_by_quadrant else 0
        rot_cells = _rotate_cells(templ_cells, size, turns)
        is_origin = int(dr) == 0 and int(dc) == 0
        off_r = int(base_r + dr * size)
        off_c = int(base_c + dc * size)
        for rr, cc in rot_cells:
            total[int(off_r + rr), int(off_c + cc)] += 1
            if is_origin:
                center[int(off_r + rr), int(off_c + cc)] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


def _occupied_output_pixels(
    templ_cells: list[tuple[int, int]],
    size: int,
    base_r: int,
    base_c: int,
    deltas: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        rot_cells = _rotate_cells(templ_cells, size, _quadrant_turns(dr, dc))
        off_r = int(base_r + dr * size)
        off_c = int(base_c + dc * size)
        for rr, cc in rot_cells:
            occupied.add((int(off_r + rr), int(off_c + cc)))
    return occupied


def _translation_only_output(grid: np.ndarray) -> np.ndarray:
    scene = _decode_scene(grid)
    deltas = [(int(dr), int(dc)) for dr, dc, _col in _normalized_deltas(scene)]
    return _render_output(
        int(scene["H"]),
        int(scene["W"]),
        int(scene["bg"]),
        int(scene["templ_col"]),
        int(scene["overlap_col"]),
        int(scene["base_r"]),
        int(scene["base_c"]),
        int(scene["size"]),
        list(scene["templ_rel"]),
        deltas,
        rotate_by_quadrant=False,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a rotated template array from an anisotropic singleton lattice.

    Input structure:
    - The background is uniform. One large monochrome component is the template, shown in its canonical orientation with color T.
    - A separated set of isolated pixels marks a coarse lattice. Exactly one singleton is also color T and serves as the lattice origin.
    - All remaining singletons are color O, and the lattice uses unequal vertical and horizontal pixel spacing.

    Rule:
    - Measure each singleton relative to the T-colored origin.
    - Infer the row lattice unit from row offsets alone and the column lattice unit from column offsets alone, then convert the raw singleton offsets into coarse lattice steps.
    - Stamp one template copy per coarse offset, translating by whole template-box sizes and rotating by quadrant: upper-right 0 degrees, upper-left 90 degrees, lower-left 180 degrees, lower-right 270 degrees.
    - Recolor the origin copy to O, and recolor any overlapped pixels to O as well.

    Key insight:
    - The singleton cluster encodes two independent things at once: anisotropic sampling tells you how to decode lattice coordinates, and once those coordinates are recovered, each offset's quadrant tells you which quarter-turn of the template belongs there.
    """
    scene = _decode_scene(grid)
    deltas = _normalized_deltas(scene)

    total = np.zeros((int(scene["H"]), int(scene["W"])), dtype=int)
    center = np.zeros((int(scene["H"]), int(scene["W"])), dtype=int)

    for dr, dc, col in deltas:
        rot_cells = _rotate_cells(list(scene["templ_rel"]), int(scene["size"]), _quadrant_turns(dr, dc))
        off_r = int(scene["base_r"] + dr * scene["size"])
        off_c = int(scene["base_c"] + dc * scene["size"])
        is_origin = int(col) == int(scene["templ_col"])
        for rr, cc in rot_cells:
            total[int(off_r + rr), int(off_c + cc)] += 1
            if is_origin:
                center[int(off_r + rr), int(off_c + cc)] += 1

    out = np.full((int(scene["H"]), int(scene["W"])), int(scene["bg"]), dtype=int)
    out[total > 0] = int(scene["templ_col"])
    out[(total == 1) & (center > 0)] = int(scene["overlap_col"])
    out[total >= 2] = int(scene["overlap_col"])
    return out


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


def _is_rotation_distinct(cells: list[tuple[int, int]], size: int) -> bool:
    seen = set()
    for turns in range(4):
        seen.add(frozenset(_rotate_cells(cells, size, turns)))
    return len(seen) == 4


def _gen_template(size: int) -> list[tuple[int, int]] | None:
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(*TEMPLATE_AREA_RANGE)
        cells = _spanning_polyomino(area, size, size)
        if cells is None:
            continue
        if _is_rotation_distinct(cells, size):
            return cells
    return None


def _gen_lattice() -> list[tuple[int, int]]:
    deltas = list(BASE_DELTAS)
    if random.randint(*SINGLETON_COUNT_RANGE) == 6:
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
    orr_lo = int(-min_dr * ROW_GUNIT)
    orr_hi = int(GRID_SIZE - 1 - max_dr * ROW_GUNIT)
    orc_lo = int(-min_dc * COL_GUNIT)
    orc_hi = int(GRID_SIZE - 1 - max_dc * COL_GUNIT)
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(base_r), int(base_r + BOX_SIZE - 1)
    tc0, tc1 = int(base_c), int(base_c + BOX_SIZE - 1)
    occupied = _occupied_output_pixels(templ_cells, BOX_SIZE, base_r, base_c, deltas)

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = int(orr + min_dr * ROW_GUNIT)
        sr1 = int(orr + max_dr * ROW_GUNIT)
        sc0 = int(orc + min_dc * COL_GUNIT)
        sc1 = int(orc + max_dc * COL_GUNIT)
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        singleton_pixels = {
            (int(orr + dr * ROW_GUNIT), int(orc + dc * COL_GUNIT))
            for dr, dc in deltas
        }
        if singleton_pixels & occupied:
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[int(base_r + tr), int(base_c + tc)] = int(templ_col)
        for dr, dc in deltas:
            rr = int(orr + dr * ROW_GUNIT)
            cc = int(orc + dc * COL_GUNIT)
            grid[rr, cc] = int(templ_col if (dr, dc) == (0, 0) else overlap_col)

        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    assert ROW_GUNIT != COL_GUNIT
    bg, templ_col, overlap_col = random.sample(range(10), 3)

    templ_cells = _gen_template(BOX_SIZE)
    if templ_cells is None:
        return None

    deltas = _gen_lattice()
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    base_r_lo = int(-min_dr * BOX_SIZE)
    base_r_hi = int(GRID_SIZE - max_dr * BOX_SIZE - BOX_SIZE)
    base_c_lo = int(-min_dc * BOX_SIZE)
    base_c_hi = int(GRID_SIZE - max_dc * BOX_SIZE - BOX_SIZE)
    if base_r_lo > base_r_hi or base_c_lo > base_c_hi:
        return None

    base_r = random.randint(base_r_lo, base_r_hi)
    base_c = random.randint(base_c_lo, base_c_hi)

    expected = _render_output(
        GRID_SIZE,
        GRID_SIZE,
        bg,
        templ_col,
        overlap_col,
        base_r,
        base_c,
        BOX_SIZE,
        templ_cells,
        deltas,
        rotate_by_quadrant=True,
    )
    if np.array_equal(
        expected,
        _render_output(
            GRID_SIZE,
            GRID_SIZE,
            bg,
            templ_col,
            overlap_col,
            base_r,
            base_c,
            BOX_SIZE,
            templ_cells,
            deltas,
            rotate_by_quadrant=False,
        ),
    ):
        return None

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
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
