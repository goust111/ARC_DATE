# Combo variant of c4d067a0 (index 26)
# combo_spec: D1/2+D9/1
# Name: 非均匀缩放+不均匀图例间距

# Design analysis
# ----------------
# concrete_behavior:
#   original:
#     - 左上角图例中的彩色单像素落在一个等距晶格上, 行步长与列步长相同。
#     - 主区域对象是同尺寸正方形, 行列方向共享一个放大步长 step。
#     - 只要找到两个已知方块与图例中的颜色对应, 就能用
#       (base_r + rr * step, base_c + cc * step) 完成全部投影。
#   combo:
#     - 图例改成“行间距 != 列间距”的稀疏晶格, 例如行每隔 3 格、列每隔 2 格。
#     - 主区域对象改成严格非正方形的实心矩形, 且 row_step != col_step。
#     - 输入里保留的两个锚点矩形来自图例中的一个对角点对, 这样它们同时暴露纵向
#       与横向位移; 求解器必须先分别恢复图例行步长 / 图例列步长, 再分别恢复主区
#       row_step / col_step, 共四个参数。
#   example_trace:
#     - 设图例像素实际位于 (1,1), (1,5), (4,1), (4,5), ... , 即 legend_row_step=3、
#       legend_col_step=4。若保留的两个矩形对应归一化图例坐标 (0,0) 与 (2,1),
#       且它们左上角位于 (6,10) 与 (16,15), 则主区步长为
#       row_step = (16 - 6) // 2 = 5, col_step = (15 - 10) // 1 = 5。
#       但若错误地把图例当成统一步长 1 或 2 的晶格, 至少一轴会得到非整数缩放,
#       因而无法合法解释样例。这正是融合项 D9/1 真正参与推理的位置。
# complexity_calibration:
#   - 图例步长只从少量档位中选取: (2,3), (2,4), (3,2), (3,4)。
#   - 主区步长也只从少量档位中选取: (4,5) 或 (5,4)。
#   - 矩形高宽都严格小于对应步长且高宽不等, 既让视觉上明显不同于原题, 又保证
#     输出矩形之间始终有背景间隙, 不会在组件检测时粘连。
# implicit_assumption_audit:
#   - 原题把 legend 的所有坐标差一起做 gcd, 默认行列共享同一图例步长; 融合后
#     必须拆成 row gcd 与 col gcd, 否则归一化坐标会在一轴上被拉伸。
#   - 原题只接受正方形主对象; 融合后必须接受任意 hh>=2, ww>=2 的实心矩形。
#   - 原题把主区行列差混合成一个统一 step; 融合后必须分开求 row_step / col_step。
#   - 原题保留同一行相邻锚点即可推断统一 step; 融合后若仍保留同排锚点, 将失去
#     纵向尺度信息, 因此生成器必须保留一个严格对角的锚点对。
# placement_legality:
#   - 采用 output-first: 先生成完整输出, 再仅保留两个锚点矩形得到输入。
#   - 通过 block_h < row_step 与 block_w < col_step 保证任意相邻投影矩形不接触。
#   - base_c 按图例最右列动态下界放置, 让主区矩形整体位于图例右侧并留出背景列。
# visual_observability:
#   - 观察样例时, 图例的纵向点距和横向点距明显不同, 主区矩形也呈现不同的纵横节奏。
#   - 两个锚点位于对角位置, 迫使解题者同时关注行和列, 而不是只看水平相邻关系。
# cleanliness:
#   - 图例仍聚集在左上角 12x12 视窗内, 主区从右侧展开, 分区干净。
#   - 输入中主区只有两个矩形, 视觉焦点仍然清晰, 不会和图例混成一团。
# color_inference:
#   - 颜色只用于把锚点矩形和图例像素对应起来, 不依赖颜色数值大小或排序。
#   - 生成器强制锚点颜色有序对在所有对角点对中唯一, 防止基于重复颜色产生多解。
# solver_generator_contract:
#   - `main()` 假设: 左上视窗内存在至少 2 个不同 legend 行与 2 个不同 legend 列,
#     且主区至少有两个同形状实心矩形。
#   - `generate_input()` 保证: 仅保留一个对角锚点对; legend_row_step != legend_col_step;
#     row_step != col_step; block_h != block_w; 并且存在“反退化”约束, 使得把图例误判
#     为单一步长时至少一轴无法得到整数主区缩放。

import random
import sys
from math import gcd
from pathlib import Path

import numpy as np

ROOT = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "solvers" / "arc_utils.py").exists():
        ROOT = _parent
        break
if ROOT is None:
    raise RuntimeError("could not locate project root containing solvers/arc_utils.py")
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg


def _axis_step(coords: list[int]) -> int:
    vals = sorted(set(int(v) for v in coords))
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    assert diffs, "legend must span at least two coordinates on each axis"
    step = diffs[0]
    for d in diffs[1:]:
        step = gcd(step, d)
    assert step > 0, "legend axis step must be positive"
    return int(step)


def _normalize_legend(legend: list[tuple[int, int, int]]) -> tuple[list[tuple[int, int, int]], int, int]:
    rs = [r for r, _, _ in legend]
    cs = [c for _, c, _ in legend]
    row_step = _axis_step(rs)
    col_step = _axis_step(cs)
    rmin = min(rs)
    cmin = min(cs)
    legend_pts = [
        ((r - rmin) // row_step, (c - cmin) // col_step, int(v))
        for r, c, v in legend
    ]
    return legend_pts, int(row_step), int(col_step)


def _dominant_shape(blocks: list[tuple[int, int, int, int, int]]) -> tuple[int, int]:
    counts: dict[tuple[int, int], int] = {}
    for _, _, hh, ww, _ in blocks:
        counts[(hh, ww)] = counts.get((hh, ww), 0) + 1
    assert counts, "must contain at least one rectangle block"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _candidate_valid(
    legend_pts: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int, int, int]],
    block_h: int,
    block_w: int,
    base_r: int,
    base_c: int,
    row_step: int,
    col_step: int,
    shape: tuple[int, int],
    grid_shape: tuple[int, int],
) -> bool:
    h, w = grid_shape
    expected: set[tuple[int, int, int]] = set()
    rects: list[tuple[int, int, int, int]] = []
    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * row_step
        tc0 = base_c + cc * col_step
        if not (0 <= tr0 <= h - block_h and 0 <= tc0 <= w - block_w):
            return False
        rect = (tr0, tc0, tr0 + block_h - 1, tc0 + block_w - 1)
        for prev in rects:
            if _rectangles_overlap(prev, rect):
                return False
        rects.append(rect)
        expected.add((tr0, tc0, col))

    for r0, c0, hh, ww, col in blocks:
        if (hh, ww) != shape:
            continue
        if (r0, c0, col) not in expected:
            return False
    return True


def _infer_mapping(
    legend_pts: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int, int, int]],
    grid_shape: tuple[int, int],
) -> tuple[int, int, int, int, int, int]:
    block_h, block_w = _dominant_shape(blocks)
    shape = (block_h, block_w)
    dom_blocks = [b for b in blocks if (b[2], b[3]) == shape]
    assert len(dom_blocks) >= 2, "must contain at least two dominant rectangle blocks"

    candidates: dict[tuple[int, int, int, int, int, int], int] = {}
    for i, (br1, bc1, _, _, col1) in enumerate(dom_blocks):
        for j, (br2, bc2, _, _, col2) in enumerate(dom_blocks):
            if i == j:
                continue
            dbr = br2 - br1
            dbc = bc2 - bc1
            if dbr <= 0 or dbc <= 0:
                continue
            for rr1, cc1, lcol1 in legend_pts:
                if lcol1 != col1:
                    continue
                for rr2, cc2, lcol2 in legend_pts:
                    if lcol2 != col2:
                        continue
                    drr = rr2 - rr1
                    dcc = cc2 - cc1
                    if drr <= 0 or dcc <= 0:
                        continue
                    if dbr % drr != 0 or dbc % dcc != 0:
                        continue
                    row_step = dbr // drr
                    col_step = dbc // dcc
                    if row_step <= 0 or col_step <= 0:
                        continue
                    base_r = br1 - rr1 * row_step
                    base_c = bc1 - cc1 * col_step
                    mapping = (base_r, base_c, row_step, col_step, block_h, block_w)
                    if _candidate_valid(
                        legend_pts,
                        dom_blocks,
                        block_h,
                        block_w,
                        base_r,
                        base_c,
                        row_step,
                        col_step,
                        shape,
                        grid_shape,
                    ):
                        candidates[mapping] = candidates.get(mapping, 0) + 1

    assert candidates, "existing blocks must determine a valid anisotropic mapping"
    max_votes = max(candidates.values())
    best = [mapping for mapping, votes in candidates.items() if votes == max_votes]
    assert len(best) == 1, "legend-to-grid mapping must be unique"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a sparse field of rectangles by expanding a top-left legend with separate row and column spacings.

    Input structure:
    - The background is the most common color.
    - In the top-left window, singleton colored pixels form a sparse legend whose row spacing and column spacing differ.
    - In the main area, a few solid rectangles of one common shape already appear; they are partial placements of that legend.

    Rule:
    - Normalize the legend using one lattice step for rows and another lattice step for columns.
    - Map each normalized legend coordinate to the main area using a shared offset plus independent row and column scaling.
    - Keep the existing rectangles and paint every missing legend position with the same solid rectangle shape and color.

    Key insight:
    - The legend is still a minimap of the large layout, but both stages are anisotropic: first the legend itself uses unequal row/column gaps, then the main area stretches those normalized coordinates by unequal row/column block steps.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int, int]] = []
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh < 2 or ww < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), int(ww), col))

    assert singles and blocks, "grid must contain legend pixels and main rectangles"

    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    legend_pts, _legend_row_step, _legend_col_step = _normalize_legend(legend)
    base_r, base_c, row_step, col_step, block_h, block_w = _infer_mapping(
        legend_pts,
        blocks,
        (h, w),
    )

    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * row_step
        tc0 = base_c + cc * col_step
        assert 0 <= tr0 <= h - block_h and 0 <= tc0 <= w - block_w, (
            "legend rectangle projection must fit in canvas"
        )
        out[tr0 : tr0 + block_h, tc0 : tc0 + block_w] = col
    return out


# --------------- generator (o5x) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_ROWS = (3, 4)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1, 2)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6
LEGEND_STEP_OPTIONS = (
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 4),
)

# === CFG: block area ===
MAIN_STEP_OPTIONS = (
    (4, 5),
    (5, 4),
)
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 700


def _enumerate_staircase_shapes():
    shapes = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue

                def rec(prefix, remaining, r):
                    if r == rows:
                        if remaining == 0 and max(prefix) == cols:
                            shapes.append(tuple(prefix))
                        return
                    left = rows - r - 1
                    lo = max(1, remaining - left * cols)
                    hi = min(cols, remaining - left)
                    for width in range(lo, hi + 1):
                        rec(prefix + [width], remaining - width, r + 1)

                rec([], count, 0)
    return shapes


def _fusion_visible(
    legend_row_step: int,
    legend_col_step: int,
    row_step: int,
    col_step: int,
) -> bool:
    shared = gcd(legend_row_step, legend_col_step)
    row_factor = legend_row_step // shared
    col_factor = legend_col_step // shared
    return (row_step % row_factor != 0) or (col_step % col_factor != 0)


_STAIRCASE_SHAPES = _enumerate_staircase_shapes()
_GEOM_OPTIONS: list[tuple[int, int, int, int, int, int]] = []
for legend_row_step, legend_col_step in LEGEND_STEP_OPTIONS:
    if legend_row_step == legend_col_step:
        continue
    for row_step, col_step in MAIN_STEP_OPTIONS:
        if row_step == col_step:
            continue
        if not _fusion_visible(legend_row_step, legend_col_step, row_step, col_step):
            continue
        for block_h in range(2, row_step):
            for block_w in range(2, col_step):
                if block_h != block_w:
                    _GEOM_OPTIONS.append(
                        (
                            legend_row_step,
                            legend_col_step,
                            block_h,
                            block_w,
                            row_step,
                            col_step,
                        )
                    )


def _pick_staircase():
    while True:
        widths = random.choice(_STAIRCASE_SHAPES)
        pts = [(r, c) for r, width in enumerate(widths) for c in range(width)]
        diag_pairs = [
            (p1, p2)
            for p1 in pts
            for p2 in pts
            if p2[0] > p1[0] and p2[1] > p1[1]
        ]
        if diag_pairs:
            return pts, diag_pairs


def _random_color_counts(total: int, k: int) -> list[int]:
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        i = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[i]))
        counts[i] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(
    pts: list[tuple[int, int]],
    anchor_pair: tuple[tuple[int, int], tuple[int, int]],
    palette: list[int],
):
    pos_set = set(pts)
    first, second = anchor_pair
    a, b = random.sample(palette, 2)
    counts = dict(zip(palette, _random_color_counts(len(pts), len(palette))))
    colors = {first: a, second: b}
    counts[a] -= 1
    counts[b] -= 1
    if counts[a] < 0 or counts[b] < 0:
        return None

    anchor_colors = (a, b)
    cells = [p for p in pts if p not in colors]

    def allowed(pos, col):
        pr, pc = pos
        for qr, qc in pos_set:
            other = (qr, qc)
            if other == pos or other not in colors:
                continue
            if qr > pr and qc > pc:
                if (pos, other) != (first, second) and (col, colors[other]) == anchor_colors:
                    return False
            if qr < pr and qc < pc:
                if (other, pos) != (first, second) and (colors[other], col) == anchor_colors:
                    return False
        return True

    def search():
        if not cells:
            return all(v == 0 for v in counts.values())
        best_i, best_opts = None, None
        for i, pos in enumerate(cells):
            opts = [col for col in palette if counts[col] > 0 and allowed(pos, col)]
            if best_opts is None or len(opts) < len(best_opts):
                best_i, best_opts = i, opts
        if not best_opts:
            return False
        pos = cells.pop(best_i)
        random.shuffle(best_opts)
        for col in best_opts:
            colors[pos] = col
            counts[col] -= 1
            if search():
                return True
            counts[col] += 1
            del colors[pos]
        cells.insert(best_i, pos)
        return False

    if not search():
        return None
    return colors


def _gen_attempt(return_meta: bool = False):
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    (
        legend_row_step,
        legend_col_step,
        block_h,
        block_w,
        row_step,
        col_step,
    ) = random.choice(_GEOM_OPTIONS)
    pts, diag_pairs = _pick_staircase()
    anchor_pair = random.choice(diag_pairs)

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    legend_bottom = LEGEND_ORIGIN[0] + max_r * legend_row_step
    legend_right = LEGEND_ORIGIN[1] + max_c * legend_col_step
    if legend_bottom >= 12 or legend_right >= 12:
        return None

    base_c_min = legend_right + 2
    r_hi = size - block_h - max_r * row_step - 1
    c_hi = size - block_w - max_c * col_step - 1
    if r_hi < BASE_R_MIN or c_hi < base_c_min:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(base_c_min, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * legend_row_step, lc0 + cc * legend_col_step] = col
        br = base_r + rr * row_step
        bc = base_c + cc * col_step
        out[br : br + block_h, bc : bc + block_w] = col

    inp = out.copy()
    keep = {anchor_pair[0], anchor_pair[1]}
    for rr, cc in pts:
        if (rr, cc) in keep:
            continue
        br = base_r + rr * row_step
        bc = base_c + cc * col_step
        inp[br : br + block_h, bc : bc + block_w] = bg

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    if return_meta:
        shared = gcd(legend_row_step, legend_col_step)
        return inp, {
            "legend_row_step": legend_row_step,
            "legend_col_step": legend_col_step,
            "block_h": block_h,
            "block_w": block_w,
            "row_step": row_step,
            "col_step": col_step,
            "anchor_pair": anchor_pair,
            "legend_size": len(pts),
            "collapse_row_factor": legend_row_step // shared,
            "collapse_col_factor": legend_col_step // shared,
            "fusion_visible": _fusion_visible(
                legend_row_step,
                legend_col_step,
                row_step,
                col_step,
            ),
        }
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
