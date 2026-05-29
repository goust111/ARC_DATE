# Variant of c4d067a0
# Dimension: D1 — 匹配条件：坐标映射方式 (match_cond)
# Change: 图例像素位置通过统一缩放因子+偏移量映射到网格位置，缩放因子由已有色块的间距与图例像素间距的比值确定... → 非均匀缩放（行列不同比例）

# Design analysis
# ----------------
# concrete_behavior:
#   original:
#     - 左上图例是规则晶格上的单像素点。
#     - 主区域中的对象是等色正方形，行列方向共享同一个步长 step。
#     - 只要知道两个已存在色块的位置，就能把图例坐标 (rr, cc) 映射到
#       (base_r + rr * step, base_c + cc * step)。
#   variant:
#     - 主区域中的对象改为等色矩形，且 height != width。
#     - 行、列方向分别使用 row_step / col_step，两者必不相同。
#     - 输入里仍只保留两个已知主区域对象，但它们不再是同一行相邻块，而是
#       图例中的一个对角点对。这样两个块同时提供纵向与横向位移，求解器必须
#       分别求出 row_step 与 col_step。
#   example_trace:
#     - 若图例中锚点位于 (0,0) 与 (2,1)，输入里对应矩形左上角位于 (6,9) 与
#       (14,14)，则原题会尝试找统一 step，使得 8 = 2*step 且 5 = 1*step，
#       这是不可能的；新规则则得到 row_step = 8//2 = 4，col_step = 5//1 = 5。
# complexity_calibration:
#   - 仍保留小图例 + 稀疏主区域的整体视觉结构，避免引入新对象种类。
#   - 只允许两档非均匀步长组合: (4,5) 或 (5,4)，既能明显区别于原题，又不会
#     因自由度过高导致样例难以归纳。
#   - 矩形边长限制为严格小于对应步长，保证块之间始终有至少 1 格背景间隙。
# implicit_assumption_audit:
#   - 原 `hh != ww` 过滤默认主对象必须是正方形；变体中必须改为接受任意
#     `hh >= 2 and ww >= 2` 的实心矩形。
#   - 原 `step = gcd(...)` 默认行列共用步长；变体中不能再把纵横差混在一起求 gcd。
#   - 原 offset 投票使用 `(br0 - rr*step, bc0 - cc*step)`；变体中必须拆成
#     `(br0 - rr*row_step, bc0 - cc*col_step)`。
#   - 原输入锚块在同一图例行，无法提供纵向尺度；变体生成器必须改为保留一个
#     对角锚点对，否则 `main()` 无法从两个块推断两个方向的尺度。
# placement_legality:
#   - 生成器先选完整输出，再删去非锚块。
#   - 所有矩形均按 `base + rr*row_step / cc*col_step` 放置，并显式检查边界。
#   - 行列步长均大于对应矩形边长，因此即使图例中相邻格都填色，输出矩形也不会
#     触碰或连通，组件检测仍稳定。
# visual_observability:
#   - 人类能直接看到主区域对象是矩形而不是正方形，并且纵向间距与横向间距不同。
#   - 两个已知锚块呈对角分布，提示需要同时匹配两个坐标方向，而不是只靠水平间距。
# cleanliness:
#   - 图例仍在左上角小窗，主区域从右侧/下方展开，和原题分区清晰。
#   - 输入中主区域只有两个矩形，不会和图例混杂。
# color_inference:
#   - 不使用颜色 id 做排序或数值变换。
#   - 颜色只承担“锚点识别”作用：生成器强制锚点颜色有序对在所有对角点对中唯一，
#     这样解题者和求解器都能从颜色对应关系定位哪两个图例点与输入矩形匹配。
# solver_generator_contract:
#   - `main()` 假设至少存在一个支配形状的实心矩形组件，以及一个位于左上窗口内的
#     单像素图例。
#   - `generate_input()` 通过构造保证: 仅两块主区域矩形保留、锚点对角且颜色对唯一、
#     row_step != col_step、block_h != block_w、所有投影矩形都合法且互不重叠。

import random
import sys
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg


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
    assert len(best) == 1, "legend-to-grid anisotropic mapping must be unique"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill in a main grid of colored rectangles so it matches a sparse legend in the top-left corner.

    Input structure:
    - The background is the most common color.
    - The top-left window contains singleton colored pixels on a small regular lattice.
    - Elsewhere there are a few solid rectangles of one common size; they are partial realizations of the legend.

    Rule:
    - Each legend pixel corresponds to one solid rectangle in the main area.
    - The mapping uses one row step and one column step, plus a shared offset; the two steps are not necessarily equal.
    - The existing rectangles identify which legend pixels are already placed, so every missing legend pixel must be projected with the same rectangle shape and anisotropic spacing.

    Key insight:
    - The top-left sprite is still a minimap of the main layout, but now rows and columns stretch by different amounts, so the solver must infer two independent scales before completing the pattern.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int, int]] = []  # (r0,c0,h,w,color)
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

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    diffs = drs + dcs
    assert diffs, "legend lattice must span at least two coordinates"
    stepL = reduce(gcd, diffs)
    assert stepL > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))

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
LEGEND_STEP = 2
LEGEND_ROWS = (3, 4)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1, 2)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6

# === CFG: block area ===
STEP_OPTIONS = ((4, 5), (5, 4))  # (row_step, col_step)
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 500


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


_STAIRCASE_SHAPES = _enumerate_staircase_shapes()
_GEOM_OPTIONS = []
for row_step, col_step in STEP_OPTIONS:
    for block_h in range(2, row_step):
        for block_w in range(2, col_step):
            if block_h != block_w:
                _GEOM_OPTIONS.append((block_h, block_w, row_step, col_step))


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


def _random_color_counts(total, k):
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


def _assign_legend_colors(pts, anchor_pair, palette):
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


def _gen_attempt(return_meta=False):
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_h, block_w, row_step, col_step = random.choice(_GEOM_OPTIONS)
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
    r_hi = size - block_h - max_r * row_step - 1
    c_hi = size - block_w - max_c * col_step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
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
        return inp, {
            "block_h": block_h,
            "block_w": block_w,
            "row_step": row_step,
            "col_step": col_step,
            "anchor_pair": anchor_pair,
            "legend_size": len(pts),
        }
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 input")
