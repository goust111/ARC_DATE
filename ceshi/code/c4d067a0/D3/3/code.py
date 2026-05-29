# Variant of c4d067a0
# Dimension: D3 — 目标位置：色块放置区域 (dst)
# Change: 网格中通过缩放因子计算出的对应位置，每个图例像素映射为一个 NxN 的实心色块，色块之间以背景色分隔... → 以重力方向堆叠放置

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis
# ===============
# spec:
#   invariant_backbone:
#     - 保留原题的两层结构: 左上角 step=2 的单像素图例 + 右侧主区域的 NxN 实心色块。
#     - 保留“已有两块锚块用于反推映射”的信息结构，但它们现在只负责确定 block_size、
#       主区域列步长 step、以及 base_c；不再用于确定 base_r。
#   d3_change_concretized:
#     - 原题里 legend 像素 (rr, cc) 映射到精确位置:
#         top_left = (base_r + rr * step, base_c + cc * step)
#     - 本变体里仍然先把每个 legend 列 cc 映射到主区域列:
#         block_col = base_c + cc * step
#       但该像素对应的色块不会去精确的 rr*step 高度，而是从该列顶部落下。
#     - 同一列中的 legend 像素按 rr 从小到大投放。较小 rr 先落下，因此会停在更低处；
#       后来的块叠在上面，最终形成无竖向间隙的竖直堆栈。
#   concrete_trace:
#     - 例: block_size=2, step=4, 某列 cc=0 在 legend 中有
#         (0,0)=红, (1,0)=蓝, (2,0)=黄
#     - 原题输出位置会是:
#         红 at row base_r+0*4, 蓝 at base_r+1*4, 黄 at base_r+2*4
#         三块之间保留背景间隔。
#     - 本变体输出位置会是:
#         红先落到底部, 蓝再落到红上方, 黄再落到蓝上方
#         三块 top-left rows 依次是 H-2, H-4, H-6, 竖向紧贴堆叠。
#     - 差异集中在“目标放置环节”; 图例识别、列对齐、块尺寸识别都延续原题骨架。
#   complexity_calibration:
#     - 只改 D3，但为了让重力效果肉眼可见，生成器要求左侧若干列有 3-4 层堆叠，
#       避免退化成仅两块平铺。
#     - 仍使用 2-4 种颜色、6-10 个 legend 像素，复杂度与原题同量级。
#     - 不额外引入新的符号、方向编码或多阶段变换，避免超过参考任务复杂度。
#   implicit_assumptions_review:
#     - 原 solver 的 base_r/base_c 投票依赖“块的行位置等于 rr*step + 常量”。
#       该前提在重力变体中失效，所以新的 solver 只从两个锚块恢复横向映射。
#     - 原 solver 末尾对全部 legend 像素直接写入固定坐标；在本变体中若继续这样做，
#       会把已有锚块重复“复制”到其上方或错误高度，所以必须改成逐块下落模拟，并跳过
#       已经存在的两个锚块。
#     - 原题中竖向 gap 保证各块分离；本变体里同列块会接触。如果上下两块颜色相同，
#       输出里会合并成一个大矩形，不利于人眼识别。生成器因此禁止同列相邻层同色。
#   placement_legality:
#     - step 仍满足 step > block_size，因此不同列之间保持背景分隔，不会横向碰撞。
#     - base_c >= 8，主区域不会与左上角 legend 列重叠。
#     - 最大列高为 4 层，且 grid 尺寸 23-27、block_size 2-4，所以竖直堆叠始终装得下。
#     - 输入只保留一对位于 row0 的相邻锚块，它们天然是各自列的最底层，因此删除其他块后
#       不会出现“悬空”块。
#   visual_observability:
#     - 从样例上可以直接看到: output 中同一主区域列会出现向上堆叠，而不是按 legend 行
#       间隔散开；这正是 D3 变体的可见变化。
#     - anchor 仍是一对相邻列的底部块，足以提示“列映射”仍然存在，但纵向行为变了。
#   main_new_assumptions:
#     - 输入恰有两个 NxN 锚块，且它们对应 legend 中唯一一对同一行、相邻列、颜色顺序匹配
#       的像素。
#     - 这两个锚块已经处于各自列的最终落点（底层）。
#     - generate_input() 会强制满足这些条件，并用 main(input)==output 反向验算。


def _drop_block(grid: np.ndarray, bg: int, block_size: int, c0: int, color: int) -> int:
    """Drop an NxN square straight down within a fixed column and paint it in-place."""
    h, w = grid.shape
    s = int(block_size)
    c0 = int(c0)
    assert 0 <= c0 <= w - s

    def clear_at(r0: int) -> bool:
        return bool(np.all(grid[r0:r0 + s, c0:c0 + s] == bg))

    r0 = 0
    assert clear_at(r0), "drop column must be clear at the top"
    while r0 + 1 <= h - s and clear_at(r0 + 1):
        r0 += 1
    grid[r0:r0 + s, c0:c0 + s] = int(color)
    return int(r0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild gravity-stacked block columns from a top-left legend and two bottom anchor blocks.

    Input structure:
    - bg = most common color.
    - Top-left corner: singleton legend pixels on a regular lattice, still encoding the small color pattern.
    - Main area: exactly two solid NxN blocks already resting in their projected columns; they reveal the block size and horizontal column spacing.

    Rule:
    - Match the two existing blocks to the unique adjacent color pair in the legend, which fixes the main-area column mapping.
    - For every other legend pixel, use its legend column to choose the destination main-area column, then drop an NxN block from the top of that column.
    - Process legend rows from top to bottom, so earlier legend rows settle lower and later rows stack above them; the two input anchor blocks stay in place.

    Key insight: The minimap still determines block colors and columns, but legend rows no longer map to absolute y-coordinates — they determine drop order inside each column.
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
        sub = g[r0:r1 + 1, c0:c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), col))

    assert singles and blocks, "grid must contain legend pixels and anchor blocks"

    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    step_l = reduce(gcd, drs + dcs)
    assert step_l > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    legend_color: dict[tuple[int, int], int] = {}
    for r, c, v in legend:
        rr = (r - rmin) // step_l
        cc = (c - cmin) // step_l
        legend_pts.append((int(rr), int(cc), int(v)))
        legend_color[(int(rr), int(cc))] = int(v)

    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert len(blocks) == 2, "gravity variant expects exactly two anchor blocks"

    blocks.sort(key=lambda b: (b[1], b[0]))
    left_block, right_block = blocks
    assert left_block[1] < right_block[1], "anchor blocks must occupy distinct columns"
    step = int(right_block[1] - left_block[1])
    assert step > block_size, "projected block columns must stay separated"

    pair_colors = (int(left_block[3]), int(right_block[3]))
    anchor_matches = []
    for rr, cc, col in legend_pts:
        right_col = legend_color.get((rr, cc + 1))
        if right_col is not None and (int(col), int(right_col)) == pair_colors:
            anchor_matches.append((int(rr), int(cc)))
    assert len(anchor_matches) == 1, "anchor blocks must match a unique adjacent legend pair"

    anchor_rr, anchor_cc = anchor_matches[0]
    base_c = int(left_block[1] - anchor_cc * step)
    assert int(right_block[1]) == base_c + (anchor_cc + 1) * step

    keep = {(anchor_rr, anchor_cc), (anchor_rr, anchor_cc + 1)}
    for cc in sorted({cc for _, cc, _ in legend_pts}):
        col_items = sorted(
            [(rr, col) for rr, ccol, col in legend_pts if ccol == cc and (rr, ccol) not in keep],
            key=lambda x: x[0],
        )
        c0 = base_c + cc * step
        assert 0 <= c0 <= w - block_size, "projected block column must fit in canvas"
        for _rr, col in col_items:
            _drop_block(out, bg, block_size, c0, col)

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
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8

# === CFG: generation ===
MAX_RETRIES = 400


def _enumerate_staircase_shapes():
    """Enumerate all valid staircase width tuples used by the legend."""
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
_ANCHORABLE_SHAPES = [widths for widths in _STAIRCASE_SHAPES if widths[0] >= 2]


def _pick_staircase():
    """Pick a staircase and a bottom-anchor pair from the top legend row."""
    widths = random.choice(_ANCHORABLE_SHAPES)
    pts = [(r, c) for r, width in enumerate(widths) for c in range(width)]
    row0_pairs = [(0, c) for c in range(widths[0] - 1)]
    assert row0_pairs
    return pts, random.choice(row0_pairs)


def _random_color_counts(total: int, k: int) -> list[int]:
    """Distribute `total` cells among `k` colors, each count in [1, MAX_COUNT_PER_COLOR]."""
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
    """Backtrack a color pattern with unique horizontal anchor pair and visible vertical layers."""
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

        up = (r - 1, c)
        if up in pos_set and up in colors and colors[up] == col:
            return False

        down = (r + 1, c)
        if down in pos_set and down in colors and colors[down] == col:
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


def _render_output(size: int, bg: int, block_size: int, step: int, base_c: int, legend_colors):
    """Render the full gravity-stacked output and record each block position."""
    out = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col

    placed = {}
    for rr, cc in sorted(legend_colors.keys()):
        c0 = base_c + cc * step
        r0 = _drop_block(out, bg, block_size, c0, legend_colors[(rr, cc)])
        placed[(rr, cc)] = (r0, c0)
    return out, placed


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    pts, anchor_pair = _pick_staircase()

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    max_c = max(c for _, c in pts)
    c_hi = size - block_size - max_c * step
    if c_hi < BASE_C_MIN:
        return None
    base_c = random.randint(BASE_C_MIN, c_hi)

    out, placed = _render_output(size, bg, block_size, step, base_c, legend_colors)

    inp = out.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for pos, (r0, c0) in placed.items():
        if pos in keep:
            continue
        inp[r0:r0 + block_size, c0:c0 + block_size] = bg

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
    raise RuntimeError("failed to generate valid c4d067a0 gravity-variant input")
