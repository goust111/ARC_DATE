# Combo variant of c4d067a0 (index 12)
# combo_spec: D7/1+D9/2
# Name: 边缘色条编码缩放+间距编码信息

from __future__ import annotations

import random
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for combo D7/1 + D9/2
# a) 具体化与行为追踪:
#    - 原题: legend 是固定 step=2 的稀疏点阵；右侧只给出两个完整方块，方块边长直接暴露 scale，
#      方块间距也天然跟着这个统一 step 走。
#    - 本融合版: 右侧仍只保留两个锚点，但它们缩水成 2x2 seed，真正的方块边长必须从边缘色条读取。
#      同时 legend 不再是等距点阵: 相邻行/列之间可能隔 2 或 3 格。这个 2/3 的差异不是噪声，
#      而是编码“对应输出方块之间留 1 格或 2 格背景”。
#    - 一个具体追踪:
#      * 若 legend 三列坐标是 c=[1,3,6]，则两段横向间距分别是 2 和 3。
#      * 边缘 bar 长度若为 4，则输出里前两列 block 的左上角差是 4+(2-1)=5，后两列差是 4+(3-1)=6。
#      * 输入里只看得到其中一对 2x2 seeds，所以只知道某一段 block 间距；其余列/行的间距必须回到
#        legend 里逐段读取，不能再假设全局统一 step。
# b) 复杂度校准:
#    - gap 只允许两档 {2,3}，视觉上明显但不会把 top-left legend 拉得过散。
#    - true scale 只允许 {3,4}，这样 seed(2x2) 与最终方块总能区分开，bar 的长度也容易读。
#    - generator 强制至少两种 gap 值都在样本中出现，并且 D7-only 的“统一 step 外推”会得到不同几何，
#      防止退化成只看 bar 就够的弱版本。
# c) 隐式假设审查:
#    - 原 `stepL = gcd(...)` 假设 legend 行列等距；这里失效，solver 改为保留真实 row/col 坐标差，
#      再把每段差值单独转成输出偏移。
#    - 原 “方块大小 == scale” 也失效；现在可见方块只是 2x2 seeds，真正 scale 来自边缘 bar。
#    - 原 “主区域只有一个统一 block step” 失效；现在只保留 base translation 是统一的，
#      而 row/col offset 要由 legend 的逐段 gap 累加得到。
#    - 原颜色投票逻辑仍可保留，但投票项从 `rr*step/cc*step` 改成 `row_offset[r] / col_offset[c]`。
# d) 放置合法性:
#    - legend 仍固定在左上、与主区域横向分离；所有 block 左上角都由单调递增 offset 生成，因此互不重叠。
#    - generator 预留最后一行和最后一列，使 bar 无论放在 bottom 还是 right 都不会碰到输出 block。
#    - 2x2 seeds 固定贴在其目标大方块的左上角，扩张时不会错位。
# e) 视觉可观察性:
#    - 人类会看到三个区域: 左上不等距 legend、边缘长条、主区域两个小 seed。
#    - 仅靠 seed 可见某一段 block 间距，无法推出全部 missing block 的摆放；仅靠 legend 又不知道方块尺寸。
#      因此样例会迫使解题者把“bar 读 scale”和“legend 读 gap”两条线索合并。
# f) 可视化整洁度:
#    - legend 仍是 staircase 形状，没有额外噪声对象；bar 只占一条边；主区域只有两枚 seed。
#    - 不同对象用空间分区区分，而不是叠在同一区域里抢解释权。
# g) 变化的可推断性:
#    - 间距编码不依赖颜色编号或隐藏排序，只依赖肉眼可见的像素间空白数量。
#    - bar 也是纯几何长度符号，不需要任何颜色语义映射。
# h) 新 solver 假设与生成保证:
#    - 存在且仅存在一条贴边单色 bar，长度即最终 scale。
#    - 存在且仅存在两个同尺寸 seed square，且二者来自同一 legend row 的相邻列。
#    - legend 的 row/col 坐标严格单调，逐段差值都在 {2,3}，并且至少一处与“统一 step 外推”不同。
#    - 颜色分配保证锚点有唯一的有序相邻颜色对；generator 会用 `main(inp)==out` 做最终验收。


# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: legend layout ===
LEGEND_ORIGIN = (1, 1)
LEGEND_ROWS = (3, 4)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1, 2)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6
LEGEND_GAP_CHOICES = (2, 3)  # actual singleton spacing inside the legend

# === CFG: block area ===
SCALE_RANGE = range(3, 5)
SEED_SIZE = 2
BASE_C_MIN = 9
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 500


def _solid_square_info(g: np.ndarray, pts: np.ndarray, bg: int):
    if len(pts) < 4:
        return None
    r0, c0, r1, c1 = bbox_coords(pts)
    hh, ww = r1 - r0 + 1, c1 - c0 + 1
    if hh != ww:
        return None
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    col = int(sub[0, 0])
    if col == bg:
        return None
    if np.all(sub == col) and sub.size == len(pts):
        return int(r0), int(c0), int(hh), col
    return None


def _find_scale_bar(g: np.ndarray, comps: list[np.ndarray], bg: int):
    h, w = g.shape
    bars = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        touches_edge = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
        if not touches_edge:
            continue
        if hh == 1 or ww == 1:
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            col = int(sub[0, 0])
            if col != bg and np.all(sub == col) and sub.size == len(pts):
                bars.append((max(hh, ww), int(r0), int(c0), int(hh), int(ww), col))
    assert bars, "grid must contain a border scale bar"
    max_len = max(item[0] for item in bars)
    bars = [item for item in bars if item[0] == max_len]
    assert len(bars) == 1, "scale bar must be unique"
    return bars[0]


def _legend_window_singles(
    singles: list[tuple[int, int, int]], h: int, w: int
) -> list[tuple[int, int, int]]:
    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"
    return legend


def _offsets_from_coords(coords: list[int], scale: int) -> tuple[dict[int, int], list[int]]:
    vals = sorted(set(int(v) for v in coords))
    assert vals, "need at least one legend coordinate"
    offsets = {vals[0]: 0}
    diffs: list[int] = []
    acc = 0
    for prev, cur in zip(vals, vals[1:]):
        diff = int(cur - prev)
        assert diff >= 2, "legend spacing must leave at least one background cell"
        diffs.append(diff)
        acc += int(scale + diff - 1)
        offsets[cur] = int(acc)
    return offsets, diffs


def _decode_legend(
    singles: list[tuple[int, int, int]], h: int, w: int, scale: int
) -> tuple[list[tuple[int, int, int]], dict[int, int], dict[int, int], list[int], list[int]]:
    legend = _legend_window_singles(singles, h, w)
    row_offsets, row_diffs = _offsets_from_coords([r for r, _, _ in legend], scale)
    col_offsets, col_diffs = _offsets_from_coords([c for _, c, _ in legend], scale)
    legend_pts: list[tuple[int, int, int]] = []
    for r, c, col in legend:
        legend_pts.append((row_offsets[int(r)], col_offsets[int(c)], int(col)))
    return legend_pts, row_offsets, col_offsets, row_diffs, col_diffs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct enlarged blocks by combining a non-uniform legend with a border scale bar.

    Input structure:
    - The background is the dominant color.
    - The top-left legend consists of isolated colored pixels, but adjacent legend rows/columns are not equally spaced: each visible gap is either 2 or 3 cells.
    - The grid edge contains one straight monochrome bar whose length encodes the final block size.
    - The main area contains exactly two small uniform seed squares marking the top-left corners of two future blocks.

    Rule:
    - Read the final block size from the edge bar.
    - Convert each successive legend row-gap / column-gap into an output offset increment: a legend gap of `d` means the next block starts `scale + (d - 1)` cells later.
    - Use the two seed squares and their colors to determine the global translation, then paint every legend pixel as a full block of the decoded size at the accumulated row/column offsets.

    Key insight:
    - The seeds reveal only where two specific blocks begin; they do not reveal the final block size and they do not determine all later gaps.
    - Solving requires fusing two clues: the edge bar provides scale, while the legend's uneven singleton spacing provides the per-step block spacing pattern.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, "grid must contain non-background objects"

    bar_len, _bar_r0, _bar_c0, _bar_h, _bar_w, _bar_col = _find_scale_bar(g, comps, bg)
    scale = int(bar_len)

    singles: list[tuple[int, int, int]] = []
    seeds: list[tuple[int, int, int, int]] = []
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        sq = _solid_square_info(g, pts, bg)
        if sq is not None:
            seeds.append(sq)

    assert singles and seeds, "grid must contain legend pixels and seed squares"
    legend_pts, _row_offsets, _col_offsets, _row_diffs, _col_diffs = _decode_legend(singles, h, w, scale)

    sizes = sorted({s for _, _, s, _ in seeds})
    assert sizes == [SEED_SIZE], "seed squares must all be 2x2"
    seeds.sort(key=lambda item: (item[0], item[1]))
    assert len(seeds) == 2, "input must contain exactly two seed squares"
    (r0a, c0a, _seed_size, _cola), (r0b, c0b, _seed_size2, _colb) = seeds
    assert r0a == r0b and c0b > c0a, "seed squares must form one horizontal anchor pair"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, _s, col in seeds:
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))

    bases: dict[tuple[int, int], int] = {}
    for roff, coff, col in legend_pts:
        if col not in by_color:
            continue
        for br0, bc0 in by_color[col]:
            base_r = int(br0 - roff)
            base_c = int(bc0 - coff)
            bases[(base_r, base_c)] = bases.get((base_r, base_c), 0) + 1
    assert bases, "seed squares must determine the legend-to-grid offset"
    max_votes = max(bases.values())
    best_bases = [base for base, votes in bases.items() if votes == max_votes]
    assert len(best_bases) == 1, "legend-to-grid offset must be unique"
    base_r, base_c = best_bases[0]

    for roff, coff, col in legend_pts:
        tr0 = int(base_r + roff)
        tc0 = int(base_c + coff)
        assert 0 <= tr0 <= h - scale and 0 <= tc0 <= w - scale, \
            "projected block must fit inside the canvas"
        out[tr0 : tr0 + scale, tc0 : tc0 + scale] = col
    return out


def _enumerate_staircase_shapes() -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue

                def rec(prefix: list[int], remaining: int, r: int) -> None:
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


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    pairs = [(r, c) for r, w in enumerate(widths) for c in range(w - 1)]
    return widths, pts, pairs


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


def _assign_legend_colors(pts, anchor_pair, palette):
    pos_set = set(pts)
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
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
        r, c = pos
        left = (r, c - 1)
        if left in pos_set and left in colors:
            if (colors[left], col) == anchor_colors and (left, pos) != (first, second):
                return False
        right = (r, c + 1)
        if right in pos_set and right in colors:
            if (col, colors[right]) == anchor_colors and (pos, right) != (first, second):
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


def _sample_bar_placement(size: int, scale: int):
    if random.choice([False, True]):
        return "bottom", random.randint(0, size - scale)
    return "right", random.randint(0, size - scale)


def _place_bar(grid: np.ndarray, bar_color: int, scale: int, placement) -> None:
    size = grid.shape[0]
    side, start = placement
    if side == "bottom":
        grid[size - 1, start : start + scale] = bar_color
    else:
        grid[start : start + scale, size - 1] = bar_color


def _build_positions(origin: int, diffs: list[int]) -> list[int]:
    vals = [int(origin)]
    cur = int(origin)
    for diff in diffs:
        cur += int(diff)
        vals.append(cur)
    return vals


def _build_offsets(scale: int, diffs: list[int]) -> list[int]:
    vals = [0]
    cur = 0
    for diff in diffs:
        cur += int(scale + diff - 1)
        vals.append(cur)
    return vals


def _sample_gap_diffs(n_steps: int) -> list[int]:
    if n_steps <= 0:
        return []
    return [int(random.choice(LEGEND_GAP_CHOICES)) for _ in range(n_steps)]


def _is_spacing_nondegenerate(
    pts: list[tuple[int, int]],
    anchor_pair: tuple[int, int],
    scale: int,
    row_diffs: list[int],
    col_diffs: list[int],
) -> bool:
    all_diffs = list(row_diffs) + list(col_diffs)
    if len(set(all_diffs)) < 2:
        return False

    row_offsets = _build_offsets(scale, row_diffs)
    col_offsets = _build_offsets(scale, col_diffs)
    anchor_gap = int(col_diffs[anchor_pair[1]] - 1)
    uniform_step = int(scale + anchor_gap)
    mismatch = 0
    for rr, cc in pts:
        if row_offsets[rr] != rr * uniform_step or col_offsets[cc] != cc * uniform_step:
            mismatch += 1
    return mismatch >= 2


def _baseline_positions(
    pts: list[tuple[int, int]],
    base_r: int,
    base_c: int,
    scale: int,
    anchor_pair: tuple[int, int],
    col_diffs: list[int],
) -> dict[tuple[int, int], tuple[int, int]]:
    anchor_gap = int(col_diffs[anchor_pair[1]] - 1)
    step = int(scale + anchor_gap)
    return {(rr, cc): (int(base_r + rr * step), int(base_c + cc * step)) for rr, cc in pts}


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    scale = random.choice(list(SCALE_RANGE))
    widths, pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)

    row_diffs = _sample_gap_diffs(len(widths) - 1)
    col_diffs = _sample_gap_diffs(max(widths) - 1)
    if not _is_spacing_nondegenerate(pts, anchor_pair, scale, row_diffs, col_diffs):
        return None

    row_positions = _build_positions(LEGEND_ORIGIN[0], row_diffs)
    col_positions = _build_positions(LEGEND_ORIGIN[1], col_diffs)
    row_offsets = _build_offsets(scale, row_diffs)
    col_offsets = _build_offsets(scale, col_diffs)

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    usable_colors = [c for c in range(NUM_COLORS) if c != bg]
    palette = random.sample(usable_colors, k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    bar_choices = [c for c in usable_colors if c not in palette]
    if not bar_choices:
        return None
    bar_color = random.choice(bar_choices)

    max_row_offset = max(row_offsets[rr] for rr, _ in pts)
    max_col_offset = max(col_offsets[cc] for _, cc in pts)
    r_hi = size - scale - max_row_offset - 1
    c_hi = size - scale - max_col_offset - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    baseline = _baseline_positions(pts, base_r, base_c, scale, anchor_pair, col_diffs)
    true_positions = {(rr, cc): (int(base_r + row_offsets[rr]), int(base_c + col_offsets[cc])) for rr, cc in pts}
    if baseline == true_positions:
        return None

    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[row_positions[rr], col_positions[cc]] = col
        br, bc = true_positions[(rr, cc)]
        out[br : br + scale, bc : bc + scale] = col

    placement = _sample_bar_placement(size, scale)
    _place_bar(out, bar_color, scale, placement)

    inp = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        inp[row_positions[rr], col_positions[cc]] = col
    _place_bar(inp, bar_color, scale, placement)

    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for rr, cc in keep:
        col = legend_colors[(rr, cc)]
        br, bc = true_positions[(rr, cc)]
        inp[br : br + SEED_SIZE, bc : bc + SEED_SIZE] = col

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
