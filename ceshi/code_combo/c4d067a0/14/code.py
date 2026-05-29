# Combo variant of c4d067a0 (index 14)
# combo_spec: D3/1+D11/0
# Name: 棋盘格背景+周期性铺贴

# Design analysis
# ----------------
# a) 具体化与行为追踪
#    - 原题左上角是稀疏单像素图例，右侧只有一组与之对应的大色块；求解时先由已有块反推 step/base，
#      再把每个图例像素补成一次性的 NxN 色块。
#    - 本融合版保留同样的图例坐标系与方块缩放，但底板改成双色棋盘格；空白不再是单色，而是按奇偶位交替。
#      同时，图例不再只对应一组块，而是定义一个宏瓦片中的局部模板，这个模板会按 period_r=tile_rows*step、
#      period_c=tile_cols*step 在主区域重复铺贴。
#    - 例子：若图例局部点为 (0,0),(0,1),(1,0)，block_size=2，step=4，宏瓦片大小为 2x2，则原题只会在
#      base+(0,0),(0,4),(4,0) 放三块；本题会在每个 base+(8k,8m) 的瓦片中重复这三块，未被块覆盖的位置仍显示棋盘格。
#
# b) 复杂度校准
#    - 仍保留 3-4 行、2-3 列的 staircase 图例与 2-4 种前景色，避免退化成过于规则的单色平铺。
#    - 生成器强制至少 2x2 个宏瓦片，让“周期性铺贴”在输入/输出中明显可见。
#    - 同时限制总前景覆盖率，确保双色棋盘格在大面积区域持续可见，背景改动不会被大块完全淹没。
#
# c) 隐式假设审查
#    - 原 solver 假设“最常见颜色就是背景”，这里失效；必须先恢复双色棋盘格的两种颜色及其奇偶相位，
#      再把偏离棋盘格的像素视为前景。
#    - 原 solver 对同色 legend 点与已有块做绝对坐标投票，这只适用于单次放置；平铺后必须对宏周期取模，
#      通过 phase=((br-rr*step) mod period_r, (bc-cc*step) mod period_c) 投票。
#    - 原 solver 默认把擦除块的位置填回单一 bg；本生成器必须把被删掉的块区域恢复成对应相位的棋盘格。
#
# d) 放置合法性
#    - 生成器先铺满整张棋盘格，再叠加图例和所有块；主区域从较右侧开始，避免与左上角图例相撞。
#    - base、reps_r、reps_c 的采样同时满足：所有完整宏瓦片都在画布内，块之间保持 gap 间隔，且至少有 4 个宏瓦片。
#    - 构造输入时只保留首个宏瓦片中的一对水平相邻锚块，其他块全部擦除并恢复棋盘格，因此不会出现重叠或残影。
#
# e) 视觉可观察性
#    - 人类可以直接看到两件事同时发生：一是背景不是纯色而是稳定的双色棋盘格，二是右侧大色块并非只补一个局部，
#      而是在多个宏瓦片里按同一局部模板重复出现。
#    - 锚块仍只给出稀疏线索，但图例、棋盘格和输出中的大范围重复结构一起构成足够强的归纳信号。
#
# f) 可视化整洁度
#    - 棋盘格仅作为底板，不引入额外边线；方块都是整齐的实心正方形，与背景在形状上容易区分。
#    - palette 明确避开两种背景色，减少“某个前景色看起来像背景延续”的视觉混淆。
#
# g) 变化的可推断性
#    - 颜色没有数值语义，前景色只表示图例槽位对应的块颜色；背景双色只表示“空位底板”。
#    - Solver 的新假设是：前景颜色不会复用两种背景色，棋盘格在未覆盖区域保持全局一致相位。
#      生成器通过颜色采样和 output-first 校验显式保证这些前提成立。

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask


def _checkerboard(h: int, w: int, even_color: int, odd_color: int) -> np.ndarray:
    rr, cc = np.indices((h, w))
    return np.where(((rr + cc) & 1) == 0, even_color, odd_color).astype(int)


def _infer_checkerboard(grid: np.ndarray) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(grid)]
    assert len(colors) >= 2, "checkerboard background needs two colors"

    h, w = grid.shape
    rr, cc = np.indices((h, w))
    parity = ((rr + cc) & 1) == 0
    even_vals = grid[parity]
    odd_vals = grid[~parity]

    even_counts = {c: int(np.count_nonzero(even_vals == c)) for c in colors}
    odd_counts = {c: int(np.count_nonzero(odd_vals == c)) for c in colors}

    scores: dict[tuple[int, int], tuple[int, int]] = {}
    for even_color in colors:
        for odd_color in colors:
            if even_color == odd_color:
                continue
            aligned = even_counts[even_color] + odd_counts[odd_color]
            misaligned = odd_counts[even_color] + even_counts[odd_color]
            scores[(even_color, odd_color)] = (aligned - misaligned, aligned)

    best_score = max(scores.values())
    best_pairs = [pair for pair, score in scores.items() if score == best_score]
    assert len(best_pairs) == 1 and best_score[0] > 0, "checkerboard phase must be identifiable"
    return best_pairs[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a tiled block pattern drawn over a two-color checkerboard background.

    Input structure:
    - The canvas background is a global checkerboard of two alternating colors rather than a single flat bg.
    - Top-left corner: singleton legend pixels on a step-2 lattice; together they describe one macro-tile's local block template.
    - Main area: a small number of solid NxN color blocks already placed on the destination lattice; these anchors sit on top of the checkerboard.

    Rule:
    - First infer the checkerboard colors/parity, then treat every cell that breaks that checkerboard as foreground.
    - Normalize the legend into lattice coordinates, infer the block size and lattice step from the anchor blocks, and infer the macro-tile phase by matching block colors against legend colors modulo the tile period.
    - For every macro-tile that fully fits in the canvas, place the legend-defined NxN blocks at the corresponding local positions, leaving all other cells as checkerboard.

    Key insight: The sparse legend is a repeating local template, not a one-off minimap, and the real background is the alternating checkerboard, so foreground must be detected by deviation from that pattern before the tiled phase can be recovered.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg_even, bg_odd = _infer_checkerboard(g)
    bg = _checkerboard(h, w, bg_even, bg_odd)
    fg_mask = g != bg

    comps = components_mask(fg_mask, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
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
    assert blocks, "must contain at least one block of the dominant size"

    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    assert step_candidates, "anchor blocks must reveal the block step"
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
    assert phases, "anchor blocks must determine the macro-tile phase"
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
                    "tiled block projection must fit in canvas"
                out[tr0:tr0 + block_size, tc0:tc0 + block_size] = col
    return out


# --------------- generator (combo D3/1 + D11/0) ---------------

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
MAX_FOREGROUND_RATIO = 0.42

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


_BLOCK_GAP_PAIRS = [
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE
    if STEP_MIN <= n + g <= STEP_MAX
]
_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, width in enumerate(widths) for c in range(width)]
    pairs = [(r, c) for r, width in enumerate(widths) for c in range(width - 1)]
    return pts, pairs


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


def _render_tiling(canvas: np.ndarray, legend_colors, block_size: int, step: int,
                   base_r: int, base_c: int, reps_r: int, reps_c: int,
                   tile_rows: int, tile_cols: int) -> None:
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

    bg_even, bg_odd = random.sample(list(range(NUM_COLORS)), 2)
    free_colors = [c for c in range(NUM_COLORS) if c not in (bg_even, bg_odd)]
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample(free_colors, k)
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
            tile_count = reps_r * reps_c
            fg_cells = tile_count * len(pts) * (block_size ** 2) + len(pts)
            if tile_count < 4 or fg_cells > int(MAX_FOREGROUND_RATIO * size * size):
                continue
            layout_options.append((reps_r, reps_c, r_max, c_max))
    if not layout_options:
        return None

    reps_r, reps_c, r_max, c_max = random.choice(layout_options)
    base_r = random.randint(BASE_R_MIN, r_max)
    base_c = random.randint(BASE_C_MIN, c_max)

    base_board = _checkerboard(size, size, bg_even, bg_odd)
    out = base_board.copy()
    lr0, lc0 = LEGEND_ORIGIN
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
                inp[br:br + block_size, bc:bc + block_size] = base_board[br:br + block_size, bc:bc + block_size]

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
    raise RuntimeError("failed to generate valid checkerboard-tiled c4d067a0 input")
