# Variant of c4d067a0
# Dimension: D4 — 变换：色块的颜色与形状 (transform)
# Change: 图例像素的颜色直接保留，放大为 NxN 的同色实心方块，无颜色变换... → 颜色映射变换

from __future__ import annotations

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis
# a) 具体化与行为追踪:
#    变体后，左上角图例仍然是 1 像素点阵楼梯形；右侧主区域仍然是按同一 lattice 放大的 NxN 方块。
#    唯一核心变化是：图例颜色不是直接复制到方块，而是先经过一个可见的 2 色映射表。
#    例: 图例相邻锚点若是 [红, 绿]，输入中的两块锚点可能变成 [蓝, 黄]。solver 先从这两块读出
#    红->蓝、绿->黄，再把所有图例点放大成对应的蓝/黄方块。
#    原题在“由颜色投票找 base”这一步依赖 legend 色与 block 色相同；变体则必须在同一步联合推断
#    “哪个 legend 邻接对对应当前 block 邻接对”以及“源色->目标色”的映射。
# b) 复杂度校准:
#    输入仍只保留 2 个锚点方块，因此若允许 3-4 种图例色，就会出现映射信息不足。为保持 D4 可解，
#    将图例色数收紧为恰好 2 种，并要求这 2 种都在锚点中出现，且都在整张图例中重复出现。
#    这样颜色变化明显，但自由度仍受控，不会因为映射不完整而产生歧义。
# c) 隐式假设审查:
#    原 solver 中 `by_color[col]` 和随后的 base 投票，隐含前提是“block 颜色 == legend 颜色”。
#    该前提在变体里失效，所以不能再按颜色直接对齐。新的 solver 只假设:
#    1) 输入保留一对同一行、左右相邻的锚点方块；
#    2) 图例中存在唯一一个具有该有序颜色对的相邻位置；
#    3) 图例全部只用这 2 种源色，因此 2 个锚点足以暴露完整映射。
# d) 放置合法性:
#    generator 继续沿用原题的 square block / 固定步长 / output-first 生成，保证:
#    - 图例像素互不接触，主区域方块彼此不重叠；
#    - 主区域与左上图例横向分离；
#    - 所有 block 投影都在画布内；
#    - 输入仅删去非锚点方块，不改变图例和两块锚点。
# e) 视觉可观察性:
#    目标方块色集合与图例色集合强制分离，因此肉眼可直接看出“不是原样放大，而是先换色再放大”。
#    同时两种图例色都会重复出现，多组方块重复使用两种目标色，便于人类从 2-3 个样例归纳映射。
# f) 可视化整洁度:
#    仍保持左上小图例 + 右侧整齐方块阵列的两区布局，不新增标记层，不引入重叠或噪声。
# g) 颜色变化可推断性:
#    颜色映射是每题内固定的一一映射，且只涉及 2 种源色与 2 种目标色。两块输入锚点恰好展示完整映射，
#    其余缺失方块只是把同一映射重复应用到别的位置，不依赖颜色 id 排序或任何不可见属性。
# h) 新 solver 假设审查:
#    `main()` 中每个候选都要求:
#    - 现有 block 恰好落在某个 legend 点经同一 `(base, step)` 投影的位置上；
#    - 该 legend 点颜色经映射后与现有 block 颜色一致；
#    - 映射覆盖所有 legend 颜色。
#    generator 通过“图例只含 2 色 + 锚点分属两色 + 邻接有序色对唯一”来保证这些假设。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a grid of enlarged blocks using a recolored legend in the top-left corner.

    Input structure:
    - The background is the most frequent color.
    - Top-left: singleton legend pixels on a regular lattice, using two source colors.
    - Main area: uniformly colored square blocks on a larger lattice; the shown anchor blocks already use remapped colors.

    Rule:
    - The legend still defines which lattice cells should receive blocks, and the visible blocks still reveal the scale and offset.
    - Unlike the original task, a legend color is not copied directly: it is converted through a per-instance color map inferred from the anchor blocks.
    - After inferring both the lattice alignment and the color map, place every missing NxN block with its mapped color.

    Key insight: the two adjacent anchor blocks jointly identify the matching adjacent legend pair, which reveals both the translation and the source-to-target color mapping.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []  # (r0, c0, size, color)
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), col))

    assert singles and blocks, "grid must contain legend pixels and main blocks"

    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    stepL = reduce(gcd, drs + dcs)
    assert stepL > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))
    legend_by_pos = {(rr, cc): col for rr, cc, col in legend_pts}
    legend_colors = sorted({col for _, _, col in legend_pts})
    assert len(legend_colors) == 2, "this variant expects exactly two legend colors"

    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert len(blocks) >= 2, "must contain at least two anchor blocks"

    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    assert step_candidates, "block step must be inferable from visible blocks"
    step = reduce(gcd, step_candidates)
    assert step > 0, "block step must be positive"

    block_by_pos = {(r0, c0): col for r0, c0, _s, col in blocks}
    legend_pairs = []
    for (rr, cc), col in sorted(legend_by_pos.items()):
        right = (rr, cc + 1)
        if right in legend_by_pos:
            legend_pairs.append((rr, cc, col, legend_by_pos[right]))
    block_pairs = []
    for (br, bc), col in sorted(block_by_pos.items()):
        right = (br, bc + step)
        if right in block_by_pos:
            block_pairs.append((br, bc, col, block_by_pos[right]))
    assert legend_pairs and block_pairs, "anchor pairs must exist in both legend and block area"

    def candidate_valid(base_r: int, base_c: int, mapping: dict[int, int]) -> bool:
        if set(mapping.keys()) != set(legend_colors):
            return False
        if len(set(mapping.values())) != len(mapping):
            return False
        for r0, c0, _s, dst_col in blocks:
            dr = r0 - base_r
            dc = c0 - base_c
            if dr < 0 or dc < 0 or dr % step != 0 or dc % step != 0:
                return False
            pos = (dr // step, dc // step)
            src_col = legend_by_pos.get(pos)
            if src_col is None or mapping.get(src_col) != dst_col:
                return False
        return True

    candidates: dict[tuple[int, int, tuple[tuple[int, int], ...]], tuple[int, int, dict[int, int]]] = {}
    for rr, cc, src_left, src_right in legend_pairs:
        if src_left == src_right:
            continue
        for br, bc, dst_left, dst_right in block_pairs:
            if dst_left == dst_right:
                continue
            mapping = {int(src_left): int(dst_left), int(src_right): int(dst_right)}
            base_r = br - rr * step
            base_c = bc - cc * step
            if candidate_valid(base_r, base_c, mapping):
                key = (
                    int(base_r),
                    int(base_c),
                    tuple(sorted((int(k), int(v)) for k, v in mapping.items())),
                )
                candidates[key] = (int(base_r), int(base_c), dict(mapping))

    assert len(candidates) == 1, "legend pair and recolor mapping must be uniquely determined"
    base_r, base_c, color_map = next(iter(candidates.values()))

    for rr, cc, src_col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, \
            "legend block projection must fit in canvas"
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = color_map[src_col]
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
LEGEND_PALETTE_SIZE = 2
MIN_COUNT_PER_COLOR, MAX_COUNT_PER_COLOR = 2, 6

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 300


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


_BLOCK_GAP_PAIRS = [
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE
    if STEP_MIN <= n + g <= STEP_MAX
]
_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    pairs = [(r, c) for r, w in enumerate(widths) for c in range(w - 1)]
    return pts, pairs


def _two_color_counts(total: int) -> list[int]:
    lo = max(MIN_COUNT_PER_COLOR, total - MAX_COUNT_PER_COLOR)
    hi = min(MAX_COUNT_PER_COLOR, total - MIN_COUNT_PER_COLOR)
    assert lo <= hi
    first = random.randint(lo, hi)
    counts = [first, total - first]
    random.shuffle(counts)
    return counts


def _assign_legend_colors(pts, anchor_pair, palette):
    pos_set = set(pts)
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    a, b = random.sample(palette, 2)
    counts = dict(zip(palette, _two_color_counts(len(pts))))
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


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)

    bg = random.randrange(NUM_COLORS)
    legend_palette = random.sample([c for c in range(NUM_COLORS) if c != bg], LEGEND_PALETTE_SIZE)
    legend_colors = _assign_legend_colors(pts, anchor_pair, legend_palette)
    if legend_colors is None:
        return None

    target_pool = [c for c in range(NUM_COLORS) if c != bg and c not in legend_palette]
    target_palette = random.sample(target_pool, LEGEND_PALETTE_SIZE)
    color_map = dict(zip(legend_palette, target_palette))

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), src_col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = src_col
        br, bc = base_r + rr * step, base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = color_map[src_col]

    inp = out.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for rr, cc in pts:
        if (rr, cc) in keep:
            continue
        br, bc = base_r + rr * step, base_c + cc * step
        inp[br : br + block_size, bc : bc + block_size] = bg

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
    raise RuntimeError("failed to generate valid c4d067a0 input")
