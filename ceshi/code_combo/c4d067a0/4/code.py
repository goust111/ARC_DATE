# Combo variant of c4d067a0 (index 4)
# combo_spec: D1/2+D7/3
# Name: 非均匀缩放+多级缩放因子

# Design analysis
# ----------------
# a) concrete_behavior:
#    - original:
#      - 左上角图例仍是 step=2 的单像素晶格，主区域是统一大小的方块。
#      - 所有图例点共用一个缩放步长 step，所有颜色也共用同一个方块边长。
#    - combo:
#      - 主区域仍按一个共享的 `(base_r + rr*row_step, base_c + cc*col_step)` 粗晶格排布，
#        但 `row_step != col_step`，因此位置映射是各向异性的。
#      - 同时，每种颜色不再共享同一个块尺寸；颜色 `col` 对应一个专属矩形
#        `(height_by_color[col], width_by_color[col])`，而且这些尺寸在行列两个方向上都可不同。
#      - 例如：若图例中红点和蓝点分别映射成 `2x5` 与 `4x3` 矩形，那么融合版不会再允许
#        它们落在一个“横纵同倍率”的布局里；生成器只允许诸如 `(row_step,col_step)=(6,5)`
#        这样的各向异性步长，并让不同颜色再额外拥有不同的 `(h,w)`，从而可见对象既不等距
#        对称，也不等形。
#      - 更具体地说，若锚点是 `(0,0)->(6,9)` 的青色 `2x5` 矩形，和 `(2,1)->(18,14)` 的
#        黄色 `4x3` 矩形，则得到 `row_step=(18-6)//2=6`、`col_step=(14-9)//1=5`；
#        之后其他青色像素要画成 `2x5`，黄色画成 `4x3`，第三种颜色例如洋红则可能画成 `3x2`。
#    - difference:
#      - 原题只需找一个 step 和一个 block_size；融合后要找一个二维位置步长
#        `(row_step,col_step)`，再为每种颜色恢复一个二维尺寸 `(h,w)`。
#
# b) complexity_calibration:
#    - 颜色数限制为 2-3，避免出现过多颜色导致尺寸规则难以概括。
#    - 位置步长只取少量档位 `(5,6)` 或 `(6,5)`，保证人类能看见明显的横纵不对称。
#    - 每个颜色至少在图例中出现 2 次，且输入里每个颜色至少保留 1 个主区域矩形样例。
#    - 选取的颜色尺寸集合必须同时覆盖至少两种高度和至少两种宽度，避免退化成
#      “只在一个方向上随颜色变化”的弱版本。
#
# c) implicit_assumption_audit:
#    - 原 `hh != ww` 过滤假设对象必须是方块；融合后必须接受任意 `hh>=2, ww>=2` 的实心矩形。
#    - 原 “dominant block size” 假设所有块同尺寸；融合后必须改成“每种颜色一个一致尺寸”。
#    - 原 `step = gcd(...)` 将行列位移混为一个数；融合后必须分别从同一对锚点推出
#      `row_step` 与 `col_step`。
#    - 原 offset 投票只依赖 `(rr*step, cc*step)`；融合后验证候选映射时还要检查
#      每个可见块的颜色尺寸是否与 `shape_by_color` 一致。
#    - 新增隐式假设是“每个图例颜色在输入里至少有一个可见矩形样例”；生成器显式保证。
#
# d) placement_legality:
#    - 生成器先构造完整输出，再删除未保留的块。
#    - 对任意颜色都要求 `height < row_step` 且 `width < col_step`，这样相邻晶格位的矩形
#      至少隔 1 格背景，不会连通或重叠。
#    - base 取值显式受边界约束，所有矩形投影都完整落在画布内，并与左上图例分区隔开。
#
# e) visual_observability:
#    - 人类能直接看到：主区域的块是矩形、横纵间距不同、不同颜色矩形大小也不同。
#    - 输入中保留一个对角锚点对，强制样例暴露行列两个方向的位移信息，而不是只暴露水平间距。
#    - 每个颜色都在输入主区域中出现过，尺寸规则有直接可见证据，不依赖颜色编号等隐藏顺序。
#
# f) cleanliness:
#    - 图例仍固定在左上小窗，主区域从右侧展开，结构分区和原题一致。
#    - 所有主区域对象都是同类的纯色矩形，没有叠加标记层；差异只体现在位置步长和尺寸上。
#
# g) inferability_of_change:
#    - 颜色不参与数值排序，也不做重映射；颜色只索引其对应的矩形形状。
#    - 由于每个颜色在图例里至少重复一次、在输入里至少给出一个主区域样例，
#      解题者可以从少量训练对中归纳出“空间映射共享，但尺寸随颜色变化，且横纵比例不同”。
#
# solver_generator_contract:
#    - `main()` 假设输入包含左上角单像素图例、至少两个位于主区域的可见矩形、
#      且每个图例颜色都已有一个可见矩形样例。
#    - `generate_input()` 保证：锚点对角且颜色有序对在所有对角点对中唯一，
#      `row_step != col_step`，颜色尺寸两轴都发生变化，所有投影合法且互不接触。

import random
import sys
from functools import reduce
from math import gcd

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, components_mask, get_bg


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _extract_objects(
    grid: np.ndarray,
    bg: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, int, int]]]:
    comps = components_mask(grid != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int, int]] = []  # (r0, c0, h, w, color)
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(grid[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh < 2 or ww < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), int(ww), col))
    return singles, blocks


def _normalize_legend(legend: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
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
    return [
        (int((r - rmin) // stepL), int((c - cmin) // stepL), int(v))
        for r, c, v in legend
    ]


def _shape_by_color(
    blocks: list[tuple[int, int, int, int, int]],
) -> dict[int, tuple[int, int]]:
    shape_map: dict[int, tuple[int, int]] = {}
    for _, _, hh, ww, col in blocks:
        shape = (int(hh), int(ww))
        if col in shape_map:
            assert shape_map[col] == shape, "each color must use one consistent rectangle shape"
        shape_map[col] = shape
    return shape_map


def _candidate_valid(
    legend_pts: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int, int, int]],
    shape_map: dict[int, tuple[int, int]],
    base_r: int,
    base_c: int,
    row_step: int,
    col_step: int,
    grid_shape: tuple[int, int],
) -> bool:
    h, w = grid_shape
    max_h = max(hh for hh, _ in shape_map.values())
    max_w = max(ww for _, ww in shape_map.values())
    if row_step <= max_h or col_step <= max_w:
        return False

    expected: dict[tuple[int, int, int], tuple[int, int]] = {}
    rects: list[tuple[int, int, int, int]] = []
    for rr, cc, col in legend_pts:
        if col not in shape_map:
            return False
        block_h, block_w = shape_map[col]
        tr0 = base_r + rr * row_step
        tc0 = base_c + cc * col_step
        if not (0 <= tr0 <= h - block_h and 0 <= tc0 <= w - block_w):
            return False
        rect = (tr0, tc0, tr0 + block_h - 1, tc0 + block_w - 1)
        for prev in rects:
            if _rectangles_overlap(prev, rect):
                return False
        rects.append(rect)
        expected[(tr0, tc0, col)] = (block_h, block_w)

    for r0, c0, hh, ww, col in blocks:
        if expected.get((r0, c0, col)) != (hh, ww):
            return False
    return True


def _infer_mapping(
    legend_pts: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int, int, int]],
    grid_shape: tuple[int, int],
) -> tuple[int, int, int, int, dict[int, tuple[int, int]]]:
    shape_map = _shape_by_color(blocks)
    legend_colors = {col for _, _, col in legend_pts}
    assert legend_colors.issubset(shape_map), "every legend color needs a visible rectangle example"
    assert len(blocks) >= 2, "must contain at least two visible rectangles"

    candidates: dict[tuple[int, int, int, int], int] = {}
    for i, (br1, bc1, _, _, col1) in enumerate(blocks):
        for j, (br2, bc2, _, _, col2) in enumerate(blocks):
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
                    if row_step <= 0 or col_step <= 0 or row_step == col_step:
                        continue
                    base_r = br1 - rr1 * row_step
                    base_c = bc1 - cc1 * col_step
                    if _candidate_valid(
                        legend_pts,
                        blocks,
                        shape_map,
                        base_r,
                        base_c,
                        row_step,
                        col_step,
                        grid_shape,
                    ):
                        key = (base_r, base_c, row_step, col_step)
                        candidates[key] = candidates.get(key, 0) + 1

    assert candidates, "existing rectangles must determine a valid fused mapping"
    max_votes = max(candidates.values())
    best = [key for key, votes in candidates.items() if votes == max_votes]
    assert len(best) == 1, "legend-to-grid fused mapping must be unique"
    base_r, base_c, row_step, col_step = best[0]
    return base_r, base_c, row_step, col_step, shape_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a grid of color-coded rectangles by matching it to a sparse legend in the top-left corner.

    Input structure:
    - The background is the most common color.
    - A small step-2 legend of singleton colored pixels sits in the top-left window.
    - In the main area, some projected legend positions already contain solid rectangles; different colors may use different rectangle heights and widths.

    Rule:
    - Every legend pixel maps onto one shared coarse lattice in the main area using a common offset plus separate row and column steps.
    - The visible rectangles reveal both parts of the rule: the anisotropic lattice spacing `(row_step, col_step)` and the color-specific rectangle shape `(height, width)`.
    - Fill every missing projected position with a solid rectangle of the legend color, using that color's own rectangle dimensions.

    Key insight:
    - The top-left pattern is still a minimap of the main layout, but the completion now factors into a global two-axis position mapping and a per-color two-axis size mapping.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    singles, blocks = _extract_objects(g, bg)
    assert singles and blocks, "grid must contain legend pixels and main rectangles"

    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"
    legend_pts = _normalize_legend(legend)

    base_r, base_c, row_step, col_step, shape_map = _infer_mapping(
        legend_pts,
        blocks,
        (h, w),
    )

    for rr, cc, col in legend_pts:
        block_h, block_w = shape_map[col]
        tr0 = base_r + rr * row_step
        tc0 = base_c + cc * col_step
        assert 0 <= tr0 <= h - block_h and 0 <= tc0 <= w - block_w, (
            "legend rectangle projection must fit in canvas"
        )
        out[tr0 : tr0 + block_h, tc0 : tc0 + block_w] = col
    return out


# --------------- generator ---------------

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
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 3
MIN_COUNT_PER_COLOR, MAX_COUNT_PER_COLOR = 2, 5

# === CFG: block area ===
STEP_OPTIONS = ((5, 6), (6, 5))
BASE_C_MIN = 8
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


_STAIRCASE_SHAPES = _enumerate_staircase_shapes()
_SHAPE_OPTIONS_BY_STEP: dict[tuple[int, int], list[tuple[int, int]]] = {}
for row_step, col_step in STEP_OPTIONS:
    options = []
    for block_h in range(2, row_step):
        for block_w in range(2, col_step):
            if block_h != block_w:
                options.append((block_h, block_w))
    _SHAPE_OPTIONS_BY_STEP[(row_step, col_step)] = options


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


def _random_color_counts(total: int, k: int):
    counts = [MIN_COUNT_PER_COLOR] * k
    remaining = total - MIN_COUNT_PER_COLOR * k
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        if not open_idxs:
            return None
        idx = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[idx]))
        counts[idx] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(pts, anchor_pair, palette):
    pos_set = set(pts)
    first, second = anchor_pair
    a, b = random.sample(palette, 2)
    count_list = _random_color_counts(len(pts), len(palette))
    if count_list is None:
        return None
    counts = dict(zip(palette, count_list))
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


def _pick_color_shapes(k: int, row_step: int, col_step: int):
    options = _SHAPE_OPTIONS_BY_STEP[(row_step, col_step)]
    for _ in range(120):
        chosen = random.sample(options, k)
        heights = {hh for hh, _ in chosen}
        widths = {ww for _, ww in chosen}
        if len(heights) >= 2 and len(widths) >= 2:
            return chosen
    return None


def _pick_visible_positions(pts, legend_colors, anchor_pair):
    keep = [anchor_pair[0], anchor_pair[1]]
    kept = set(keep)
    anchor_colors = {legend_colors[anchor_pair[0]], legend_colors[anchor_pair[1]]}
    for col in sorted(set(legend_colors.values()) - anchor_colors):
        candidates = [p for p in pts if legend_colors[p] == col and p not in kept]
        candidates.sort(key=lambda p: (abs(p[0] - anchor_pair[0][0]), abs(p[1] - anchor_pair[0][1])))
        assert candidates, "every non-anchor color must have another position"
        keep.append(candidates[0])
        kept.add(candidates[0])
    return keep


def _gen_attempt(return_meta=False):
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    pts, diag_pairs = _pick_staircase()
    anchor_pair = random.choice(diag_pairs)

    bg = random.randrange(NUM_COLORS)
    max_palette = min(PALETTE_SIZE_MAX, len(pts), len(pts) // MIN_COUNT_PER_COLOR)
    if max_palette < PALETTE_SIZE_MIN:
        return None
    k = random.randint(PALETTE_SIZE_MIN, max_palette)
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    row_step, col_step = random.choice(STEP_OPTIONS)
    color_shapes = _pick_color_shapes(k, row_step, col_step)
    if color_shapes is None:
        return None
    shape_by_color = dict(zip(palette, color_shapes))

    max_h = max(hh for hh, _ in shape_by_color.values())
    max_w = max(ww for _, ww in shape_by_color.values())
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - max_h - max_r * row_step
    c_hi = size - max_w - max_c * col_step
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        block_h, block_w = shape_by_color[col]
        br = base_r + rr * row_step
        bc = base_c + cc * col_step
        out[br : br + block_h, bc : bc + block_w] = col

    inp = out.copy()
    keep = set(_pick_visible_positions(pts, legend_colors, anchor_pair))
    for rr, cc in pts:
        if (rr, cc) in keep:
            continue
        col = legend_colors[(rr, cc)]
        block_h, block_w = shape_by_color[col]
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
            "row_step": row_step,
            "col_step": col_step,
            "shape_by_color": shape_by_color,
            "palette": palette,
            "anchor_pair": anchor_pair,
            "visible_count": len(keep),
        }
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
