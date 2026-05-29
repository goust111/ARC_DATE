# Combo variant of c4d067a0 (index 18)
# combo_spec: D4/0+D8/2
# Name: 颜色映射变换+逆向推断图例

from __future__ import annotations

import random
import sys
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis
# a) 具体化与行为追踪:
#    原题是“左上 legend 给出小像素图，右侧只露出少量大方块，要把缺块补全”。
#    本融合版把两条信息流都反过来:
#    - D8/2: 输入里的右侧大方块矩阵已经完整，左上 legend 被清空；
#    - D4/0: 这些大方块还不是原色，而是经过固定颜色对换后的目标色。
#    一个具体追踪:
#    - 输出真实规则: legend[(0,0)]=红、legend[(0,1)]=灰、legend[(1,0)]=蓝，
#      对应的大方块会显示成蓝、粉、红。
#    - 输入时，右侧蓝/粉/红方块完整保留，左上 legend 整块擦成背景。
#    - 求解时，先把每个实心 NxN 方块缩回一个 lattice cell，恢复占用形状；
#      再把颜色按固定对换表求逆，蓝->红、粉->灰、红->蓝，把原色 legend 重新画回左上角。
#    和原题相比，位置推理从“legend 放大到 blocks”变为“blocks 缩回 legend”，
#    颜色推理从“直接复制颜色”变为“对可见目标色做逆映射”。
# b) 复杂度校准:
#    若颜色映射是每题任意变化、同时 legend 又完全缺失，那么单题输入无法提供足够线索。
#    因此这里把映射收紧为“跨样例一致的固定对换表”，让训练样例暴露稳定的颜色规律；
#    同时每题只用 2-4 种颜色，且每种源色至少重复 2 次，避免颜色过多或只出现 1 次造成弱信号。
#    这样融合后的差异非常明显: 输入里只有完整的大方块阵列，输出只恢复左上角的小图例，
#    而且图例颜色系统性地不同于方块颜色。
# c) 隐式假设审查:
#    原 solver 隐含假设“legend 单像素存在且颜色与 block 相同”，这两点都失效了:
#    - `singles` 不能再作为必要输入，因为 legend 可以完全为空；
#    - `by_color[col]` 的同色对齐逻辑不能再使用，因为 block 色是变换后的目标色。
#    新 solver 改成依赖以下前提:
#    - 输入存在一组完整、等大、同 lattice 对齐的实心方块；
#    - 这些方块的最小 top-left 就是逻辑坐标 (0,0) 的投影；
#    - 所有可见 block 颜色都来自固定对换表，可被唯一逆回原色。
#    generator 会保证这些前提成立，并且不会混入别的非方块前景对象。
# d) 放置合法性:
#    继续沿用原题的 staircase legend 家族、LEGEND_ORIGIN=(1,1)、LEGEND_STEP=2、
#    block_size 2-4、step 3-5、base_c>=8。这样 legend 区域和 block 区域横向分离，
#    所有方块互不重叠、都在边界内，清空 legend 时也不会影响主区域。
# e) 视觉可观察性:
#    人类看到训练样例时会观察到两件稳定事实:
#    1) 输入右侧总是完整的大方块矩阵，而输出只在左上恢复一个稀疏小图例；
#    2) 相同颜色对换在不同样例里反复出现，例如蓝块总缩回红像素、粉块总缩回灰像素。
#    这足以支持“先缩回位置，再反着换色”的归纳。
# f) 可视化整洁度:
#    画面仍只有两个清晰区域: 左上角空白/恢复中的 legend 窗口，和右侧规整的 block 阵列。
#    没有叠层标记、没有重叠对象，也没有依赖细碎噪声来编码规则。
# g) 变化的可推断性:
#    颜色变化不是任意置换，而是全任务共享的一组固定成对替换。每个样例只抽其中 2-4 对里的若干对，
#    且每种颜色有重复出现，因此解题者能从少量训练例看到“某种 block 色总会变回哪种 legend 色”。
#    该规律完全基于可见颜色本身，不依赖颜色编号排序或其他不可见属性。
# h) 新 solver 假设与生成保证:
#    - `step` 来自 block top-left 正差分的 gcd，因此 block 集必须跨至少两行两列；
#      staircase 形状天然满足这一点。
#    - `base_r/base_c = min(top-left)` 依赖逻辑形状包含 (0,0)；楼梯形枚举始终包含该点。
#    - `FIXED_RECOLOR[target]` 依赖目标色来自已知对换表；generator 只从这些对换里选色。
#    - 图例清空后 output 仍可唯一恢复，依赖 block 矩阵是完整的；generator 从 output-first 构造，
#      输入只删除 legend，不删除任何 block。


# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: fixed recolor pairs ===
COLOR_SWAP_PAIRS = (
    (0, 9),
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
)
FIXED_RECOLOR: dict[int, int] = {}
for _a, _b in COLOR_SWAP_PAIRS:
    FIXED_RECOLOR[int(_a)] = int(_b)
    FIXED_RECOLOR[int(_b)] = int(_a)

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_ROWS = (3, 4)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1, 2)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MIN_COUNT_PER_COLOR, MAX_COUNT_PER_COLOR = 2, 6

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 300


def _positive_gcd(values: list[int]) -> int:
    vals = [int(v) for v in values if int(v) > 0]
    assert vals, "need at least one positive difference"
    return reduce(gcd, vals)


def _extract_blocks(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    blocks: list[tuple[int, int, int, int]] = []
    for pts in components_mask(grid != bg, connectivity=4):
        if len(pts) <= 1:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), int(col)))
    return blocks


def _block_lattice(
    blocks: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int, dict[tuple[int, int], int]]:
    sizes = [size for _, _, size, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert blocks, "must contain at least one block of the dominant block size"

    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    step = _positive_gcd(step_candidates)

    base_r = min(tops)
    base_c = min(lefts)
    block_cells: dict[tuple[int, int], int] = {}
    for r0, c0, _size, col in blocks:
        dr = int(r0) - base_r
        dc = int(c0) - base_c
        assert dr % step == 0 and dc % step == 0, "blocks must align to one lattice"
        rr = dr // step
        cc = dc // step
        pos = (int(rr), int(cc))
        prev = block_cells.get(pos)
        assert prev is None or prev == col, "lattice cells must not conflict"
        block_cells[pos] = int(col)
    return block_size, step, base_r, base_c, block_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a missing top-left legend by shrinking a recolored matrix of enlarged blocks.

    Input structure:
    - The background is the most common color.
    - Top-left: the fixed legend lattice is empty or nearly empty.
    - Main area: every occupied legend cell already appears as a solid NxN block, but block colors use a transformed palette rather than the legend's original colors.

    Rule:
    - Recover the block lattice from the complete matrix of aligned square blocks.
    - Treat each block as one enlarged legend cell and convert its visible color back through the inverse of the task's fixed recolor table.
    - Redraw the miniature legend on the top-left lattice while leaving the enlarged blocks unchanged.

    Key insight: the block matrix now carries the full spatial pattern, so solving means shrinking it back to the legend grid and undoing the consistent color transformation on each sampled block.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    blocks = _extract_blocks(g, bg)
    assert blocks, "grid must contain enlarged blocks"

    _block_size, _step, _base_r, _base_c, block_cells = _block_lattice(blocks)
    assert block_cells, "block lattice must define at least one legend cell"

    used_targets = {int(col) for col in block_cells.values()}
    assert all(col in FIXED_RECOLOR for col in used_targets), "block colors must be invertible"

    max_rr = max(rr for rr, _ in block_cells)
    max_cc = max(cc for _, cc in block_cells)
    lr0, lc0 = LEGEND_ORIGIN

    for rr in range(max_rr + 1):
        for cc in range(max_cc + 1):
            tr = lr0 + rr * LEGEND_STEP
            tc = lc0 + cc * LEGEND_STEP
            assert 0 <= tr < h and 0 <= tc < w, "legend lattice must fit inside the grid"
            out[tr, tc] = bg

    for (rr, cc), target_col in block_cells.items():
        tr = lr0 + rr * LEGEND_STEP
        tc = lc0 + cc * LEGEND_STEP
        out[tr, tc] = int(FIXED_RECOLOR[int(target_col)])

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


_BLOCK_GAP_PAIRS = [
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE if STEP_MIN <= n + g <= STEP_MAX
]
_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase() -> list[tuple[int, int]]:
    widths = random.choice(_STAIRCASE_SHAPES)
    return [(r, c) for r, width in enumerate(widths) for c in range(width)]


def _random_color_counts(total: int, k: int) -> list[int]:
    assert total >= k * MIN_COUNT_PER_COLOR
    counts = [MIN_COUNT_PER_COLOR] * k
    remaining = total - k * MIN_COUNT_PER_COLOR
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        idx = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[idx]))
        counts[idx] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _pick_source_palette(bg: int, total_pts: int) -> list[int]:
    available_pairs = [pair for pair in COLOR_SWAP_PAIRS if bg not in pair]
    k_hi = min(PALETTE_SIZE_MAX, len(available_pairs), total_pts // MIN_COUNT_PER_COLOR)
    assert PALETTE_SIZE_MIN <= k_hi
    chosen_pairs = random.sample(available_pairs, random.randint(PALETTE_SIZE_MIN, k_hi))

    palette: list[int] = []
    for a, b in chosen_pairs:
        if random.choice((False, True)):
            palette.append(int(a))
        else:
            palette.append(int(b))
    return palette


def _assign_legend_colors(
    pts: list[tuple[int, int]], palette: list[int]
) -> dict[tuple[int, int], int]:
    counts = _random_color_counts(len(pts), len(palette))
    pool: list[int] = []
    for col, count in zip(palette, counts):
        pool.extend([int(col)] * int(count))
    random.shuffle(pool)
    colors = {pos: pool[i] for i, pos in enumerate(pts)}
    assert set(colors.values()) == set(palette), "all source colors must appear"
    return colors


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    pts = _pick_staircase()

    bg = random.randrange(NUM_COLORS)
    source_palette = _pick_source_palette(bg, len(pts))
    legend_colors = _assign_legend_colors(pts, source_palette)
    color_map = {int(src): int(FIXED_RECOLOR[int(src)]) for src in source_palette}

    used_sources = set(source_palette)
    used_targets = {color_map[src] for src in source_palette}
    assert used_sources.isdisjoint(used_targets), "source and target palettes must differ"

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    out = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), source_col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = int(source_col)
        br = base_r + rr * step
        bc = base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = int(color_map[source_col])

    inp = out.copy()
    for rr in range(max_r + 1):
        for cc in range(max_c + 1):
            inp[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = bg

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
