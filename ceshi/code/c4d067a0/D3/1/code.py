# Variant of c4d067a0
# Dimension: D3 — 目标位置：色块放置区域 (dst)
# Change: 网格中通过缩放因子计算出的对应位置，每个图例像素映射为一个 NxN 的实心色块，色块之间以背景色分隔... → 铺贴到周期性网格的指定单元

# Design analysis
# ----------------
# a) 具体化与行为追踪
#    - 左上角图例仍然是 step=2 的单像素点阵，颜色和原题一致，表示一个小的“模板”。
#    - 右侧主区域仍然由 NxN 实心色块组成，色块顶点落在步长为 step=block_size+gap 的块格点上。
#    - 变体的核心改动：图例不再只映射到主区域中的一个单次形状，而是作为“一个宏瓦片内部的局部占位图”。
#      设图例在格点坐标中的包围盒是 tile_rows x tile_cols，则主区域的宏瓦片周期就是
#      period_r = tile_rows * step, period_c = tile_cols * step。每个瓦片里，图例中的每个点都映射为一个 NxN 色块。
#    - 小例子：
#      原题若图例局部坐标有 (0,0),(0,1),(1,0)，并从已有块反推出 base=(4,10), step=3，
#      则只会在 (4,10),(4,13),(7,10) 放三块。
#      本变体中，若图例包围盒是 2x2，则 period=(6,6)；除了 base 对应的那一组外，
#      还会在 (4+6k,10+6m) 的每个瓦片里重复放置同样的三块，直到完整瓦片放不下为止。
#
# b) 复杂度校准
#    - 仍保留原题的 2-4 种颜色、3-4 x 2-3 的小图例、2x2 以上实心色块，避免退化成过于规则的单色平铺。
#    - 为了保证“周期性”肉眼可见，生成器强制至少 2x2 个宏瓦片，总块数严格大于单次图例块数。
#    - 仍只保留两个输入锚块，但要求它们对应图例中唯一的水平相邻有序颜色对，这样既维持原题的稀疏提示，
#      又能唯一锁定宏瓦片的相位。
#
# c) 隐式假设审查（相对原 main 的关键变化）
#    - 原代码把 base_r + rr*step / base_c + cc*step 当成唯一绝对目标位置；本变体不成立，
#      因为同一图例点会在多个瓦片中重复出现，所以 solver 必须改为推断“模周期的相位”而不是单次绝对偏移。
#    - 原代码对同色 legend 点与 block 的投票是绝对坐标投票；本变体改为对
#      ((br0 - rr*step) mod period_r, (bc0 - cc*step) mod period_c) 投票。
#    - 原代码最后只绘制一轮 legend_pts；本变体必须在所有完整瓦片上循环绘制。
#
# d) 放置合法性
#    - 生成器只接受完整瓦片可放置的布局：tile_span_r/tile_span_c 给出单个瓦片的实际占用范围，
#      base 与 repetitions 的选择保证所有块都在画布内，且与左上角图例分离。
#    - 输入只保留同一瓦片中的两个水平相邻锚块，因此它们既能给出 step，也不会与其他重复实例混淆。
#
# e) 视觉可观察性
#    - 人类从样例能直接看到：左上角是一个小模板，右侧不是补一个局部缺块，而是整齐地在多块大瓦片中重复。
#    - 重复次数固定至少 4 个瓦片，因此 D3 的差异不是边缘现象。
#
# f) 可视化整洁度
#    - 宏瓦片之间只通过背景色间隔，没有额外标记线，视觉上会形成干净的重复阵列。
#    - 图例保留在左上角，主区域从较右侧开始，避免与平铺区混在一起。
#
# g) 颜色变化可推断性
#    - 颜色仍只表示“这个局部槽位要填什么色”，没有引入任何依赖颜色 id 的排序或运算。
#    - 锚块的唯一性由“局部相邻颜色对”保证，而不是由颜色大小关系保证。

import numpy as np
from functools import reduce
from math import gcd

from arc_utils import bbox_coords, components_mask, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat a legend-defined block template across a periodic macro-tile grid.

    Input structure:
    - bg = most common color.
    - Top-left corner: a sparse legend of singleton colored pixels on a regular lattice; these pixels describe one macro-tile's local block layout.
    - Main area: a few uniformly colored square blocks (NxN) already placed on the same block lattice; they reveal block size, block step, and macro-tile phase.

    Rule:
    - Normalize the legend into lattice coordinates, and treat its bounding box as the row/column span of one repeating macro-tile.
    - Infer the block lattice step from the existing blocks, then infer the macro-tile phase by matching block colors against legend colors modulo the macro-tile period.
    - For every macro-tile that fully fits in the canvas, place an NxN block for every legend pixel at the corresponding local position inside that tile.

    Key insight: The legend is no longer a one-off minimap of a single block cluster; it is a local template that repeats at a fixed 2D period across the destination lattice.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []  # (r0,c0,size,color)
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = g[r0:r1 + 1, c0:c1 + 1]
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

    tile_rows = max(rr for rr, _, _ in legend_pts) + 1
    tile_cols = max(cc for _, cc, _ in legend_pts) + 1
    max_rr = tile_rows - 1
    max_cc = tile_cols - 1

    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert blocks, "must contain at least one block of the dominant block size"

    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    assert step_candidates, "existing blocks must reveal the block step"
    step = reduce(gcd, step_candidates)
    assert step > 0, "block step must be positive"

    period_r = tile_rows * step
    period_c = tile_cols * step
    tile_span_r = max_rr * step + block_size
    tile_span_c = max_cc * step + block_size

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, _, col in blocks:
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))

    phases: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        if col not in by_color:
            continue
        for br0, bc0 in by_color[col]:
            phase_r = (br0 - rr * step) % period_r
            phase_c = (bc0 - cc * step) % period_c
            phases[(phase_r, phase_c)] = phases.get((phase_r, phase_c), 0) + 1
    assert phases, "existing blocks must determine the macro-tile phase"
    max_votes = max(phases.values())
    best_phases = [phase for phase, votes in phases.items() if votes == max_votes]
    assert len(best_phases) == 1, "macro-tile phase must be unique"
    phase_r, phase_c = best_phases[0]

    row_starts = range(phase_r, h - tile_span_r + 1, period_r)
    col_starts = range(phase_c, w - tile_span_c + 1, period_c)
    for tile_r in row_starts:
        for tile_c in col_starts:
            for rr, cc, col in legend_pts:
                tr0 = tile_r + rr * step
                tc0 = tile_c + cc * step
                assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, \
                    "tiled legend block projection must fit in canvas"
                out[tr0:tr0 + block_size, tc0:tc0 + block_size] = col
    return out


# --------------- generator (o5x -> D3 tiling) ---------------

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
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8
BASE_R_MIN = 1
TILE_REPS_R = (2, 3)
TILE_REPS_C = (2, 3, 4)

# === CFG: generation ===
MAX_RETRIES = 400


def _enumerate_staircase_shapes():
    """Enumerate valid staircase width tuples."""
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
    """Pick a staircase legend shape; return filled points and horizontal adjacent pairs."""
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, width in enumerate(widths) for c in range(width)]
    pairs = [(r, c) for r, width in enumerate(widths) for c in range(width - 1)]
    return pts, pairs


def _random_color_counts(total, k):
    """Distribute `total` items among `k` colors with positive counts."""
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
    """
    Color the legend so the ordered horizontal anchor pair is unique.

    This preserves the original task's sparse-anchors feel while giving the
    tiling solver a unique phase vote.
    """
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


def _render_tiling(canvas, legend_colors, block_size, step, base_r, base_c, reps_r, reps_c, tile_rows, tile_cols):
    """Paint every block in every macro-tile."""
    period_r = tile_rows * step
    period_c = tile_cols * step
    for tile_idx_r in range(reps_r):
        tile_r = base_r + tile_idx_r * period_r
        for tile_idx_c in range(reps_c):
            tile_c = base_c + tile_idx_c * period_c
            for (rr, cc), col in legend_colors.items():
                br = tile_r + rr * step
                bc = tile_c + cc * step
                canvas[br:br + block_size, bc:bc + block_size] = col


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    tile_rows = max(r for r, _ in pts) + 1
    tile_cols = max(c for _, c in pts) + 1
    tile_span_r = (tile_rows - 1) * step + block_size
    tile_span_c = (tile_cols - 1) * step + block_size
    period_r = tile_rows * step
    period_c = tile_cols * step

    layout_options = []
    for reps_r in TILE_REPS_R:
        total_h = tile_span_r + (reps_r - 1) * period_r
        r_hi = size - total_h
        r_max = min(period_r - 1, r_hi)
        if r_max < BASE_R_MIN:
            continue
        for reps_c in TILE_REPS_C:
            total_w = tile_span_c + (reps_c - 1) * period_c
            c_hi = size - total_w
            c_max = min(period_c - 1, c_hi)
            if c_max < BASE_C_MIN:
                continue
            if reps_r * reps_c < 4:
                continue
            layout_options.append((reps_r, reps_c, r_max, c_max))
    if not layout_options:
        return None

    reps_r, reps_c, r_max, c_max = random.choice(layout_options)
    base_r = random.randint(BASE_R_MIN, r_max)
    base_c = random.randint(BASE_C_MIN, c_max)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
    _render_tiling(out, legend_colors, block_size, step, base_r, base_c, reps_r, reps_c, tile_rows, tile_cols)

    inp = out.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for tile_idx_r in range(reps_r):
        tile_r = base_r + tile_idx_r * period_r
        for tile_idx_c in range(reps_c):
            tile_c = base_c + tile_idx_c * period_c
            for rr, cc in pts:
                if tile_idx_r == 0 and tile_idx_c == 0 and (rr, cc) in keep:
                    continue
                br = tile_r + rr * step
                bc = tile_c + cc * step
                inp[br:br + block_size, bc:bc + block_size] = bg

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
    raise RuntimeError("failed to generate valid tiled c4d067a0 input")
