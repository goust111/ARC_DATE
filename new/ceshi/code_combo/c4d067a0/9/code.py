# Combo variant of c4d067a0 (index 9)
# combo_spec: D5/0+D6/0
# Name: 图例消失+紧凑裁剪

from __future__ import annotations

import random
import sys
from functools import reduce
from math import gcd

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for combo D5/0 + D6/0
# a) 具体化与行为追踪:
#    原题 input 左上有完整 legend，主区域只有 2 个已知放大块；output 仍保留整张原画布，
#    只是把剩余块补齐。这里把最后两步绑定起来:
#    1) 先像原题一样，根据 legend + 两个锚块推断 block_size / step / base。
#    2) 再把所有 legend 像素都投影成放大块。
#    3) 但结果不再保留左上 legend，而是只保留所有放大块的外接矩形。
#    一个具体追踪:
#    - 原题: legend 在 (1,1),(1,3),(3,1)...；主区已有两个 3x3 锚块；求解后输出还是 25x25，
#      左上小像素仍在，右侧多出缺失块。
#    - 融合后: 同样先补全右侧全部 3x3 块，但最终答案直接变成例如 11x8 的块区域裁剪图，
#      左上 legend 与四周空白都消失。
#    差异不只是“把 legend 擦掉”或“最后 crop 一下”各做各的，而是 D5/0 先改变哪些像素算
#    输出主体，D6/0 再只围绕这个新主体裁剪；若不先删 legend，tight crop 会把左上角也一起保留。
# b) 复杂度校准:
#    仍保留原题的少量离散自由度: legend 维持 3-4 行、2-3 列的 staircase，颜色 2-4 种，
#    block_size 为 2-4，step 为 3-5，输入只给一对相邻锚块。这样“从两个块反推缩放与偏移”
#    的核心难度不变，而输出尺寸明显缩小，肉眼能直接看出与原题不同。
# c) 隐式假设审查:
#    - 原 `out = g.copy()` 隐含“原画布就是输出画布”，D6/0 后失效；新 solver 改为直接构造块区域输出。
#    - 原 `return out` 隐含“legend 应留在答案里”，D5/0 后失效；现在 legend 只作为中间指令使用。
#    - 原 legend 提取依赖顶部左侧存在多个 singleton，这一点仍成立，因为 D5/0 只改 output。
#    - 原 block 投票依赖两个锚块给出唯一 base；这点不变，generator 继续用唯一相邻色对保证。
#    - 原 `step` 由 block top-left 差分 gcd 得到；输入只有同一行的两个锚块时，仍至少有一个正列差，
#      所以 gcd 仍可求。
# d) 放置合法性:
#    generator 继续 output-first：先选完整 legend 与完整 block 布局，再删成 input。
#    所有 block 用同一 `(base_r, base_c)`、同一 `step` 投影，彼此不会重叠；`r_hi/c_hi` 预留底边和右边
#    margin，`BASE_C_MIN` 让主区远离 legend，保证不越界、不碰撞。
# e) 视觉可观察性:
#    看几组样例时，人能明显看到: input 左上有小图例、右侧只有两块；output 却只有一块紧凑的放大方块区域。
#    因为 output 完全不含 legend，解题者会自然归纳“图例只是说明书，不属于答案”。
# f) 可视化整洁度:
#    输入仍保持左上稀疏 legend 与右侧规整方块区两块区域，结构清楚。输出只剩规整块阵列的裁剪框，
#    没有重叠层或额外噪声，视觉上比原题更干净。
# g) 变化的可推断性:
#    本融合不引入新的颜色映射，只改变哪些区域保留到输出，因此不会出现颜色规则不可归纳的问题。
# h) 新 solver 假设与生成保证:
#    - 假设: 所有非 singleton 大组件里，真正有用的是等大的实心正方形。保证: generator 只生成这种块。
#    - 假设: legend 在左上 12x12 小窗内。保证: 继续使用固定 `(1,1)` 原点和 step=2。
#    - 假设: 投影后的块外接框非空。保证: staircase 至少有 6 个填充格，因此总有块。
#    - 假设: 融合不退化。保证: generator 额外拒绝“未擦除 legend 就 tight crop”也得到相同输出的样本。


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

# === CFG: generation ===
MAX_RETRIES = 300


def _positive_gcd(values: list[int]) -> int:
    vals = [int(v) for v in values if int(v) > 0]
    assert vals, "need at least one positive difference"
    return reduce(gcd, vals)


def _extract_components(
    grid: np.ndarray, bg: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
    for pts in components_mask(grid != bg, connectivity=4):
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(grid[r, c])))
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
            blocks.append((int(r0), int(c0), int(hh), col))
    return singles, blocks


def _extract_legend(
    singles: list[tuple[int, int, int]], h: int, w: int
) -> list[tuple[int, int, int]]:
    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, col) for (r, c, col) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    step_l = _positive_gcd(drs + dcs)
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, col in legend:
        rr = (r - rmin) // step_l
        cc = (c - cmin) // step_l
        legend_pts.append((int(rr), int(cc), int(col)))
    return legend_pts


def _dominant_blocks(
    blocks: list[tuple[int, int, int, int]]
) -> tuple[int, list[tuple[int, int, int, int]]]:
    sizes = [size for _, _, size, _ in blocks]
    size_vals, size_cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(size_vals[size_cnts.argmax()])
    filtered = [b for b in blocks if b[2] == block_size]
    assert filtered, "must contain at least one block of the dominant block size"
    return block_size, filtered


def _infer_step(blocks: list[tuple[int, int, int, int]]) -> int:
    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    return _positive_gcd(step_candidates)


def _infer_base(
    legend_pts: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int, int]],
    step: int,
) -> tuple[int, int]:
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, _size, col in blocks:
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))

    bases: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        if col not in by_color:
            continue
        for br0, bc0 in by_color[col]:
            base_r = br0 - rr * step
            base_c = bc0 - cc * step
            bases[(int(base_r), int(base_c))] = bases.get((int(base_r), int(base_c)), 0) + 1

    assert bases, "existing blocks must determine the legend-to-grid offset"
    max_votes = max(bases.values())
    best_bases = [base for base, votes in bases.items() if votes == max_votes]
    assert len(best_bases) == 1, "legend-to-grid offset must be unique"
    return best_bases[0]


def _render_cropped_blocks(
    legend_pts: list[tuple[int, int, int]],
    *,
    bg: int,
    block_size: int,
    step: int,
    base_r: int,
    base_c: int,
) -> np.ndarray:
    placed: list[tuple[int, int, int]] = []
    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        placed.append((int(tr0), int(tc0), int(col)))

    assert placed, "there must be at least one projected block"
    r0 = min(tr0 for tr0, _, _ in placed)
    c0 = min(tc0 for _, tc0, _ in placed)
    r1 = max(tr0 + block_size - 1 for tr0, _, _ in placed)
    c1 = max(tc0 + block_size - 1 for _, tc0, _ in placed)

    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for tr0, tc0, col in placed:
        rr0 = tr0 - r0
        cc0 = tc0 - c0
        out[rr0 : rr0 + block_size, cc0 : cc0 + block_size] = int(col)
    return out


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != bg)
    assert nz.size > 0, "grid must contain non-background pixels"
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the enlarged block pattern from the corner legend, then output only that pattern.

    Input structure:
    - The background is the grid's most frequent color.
    - A sparse single-pixel legend sits in the upper-left on a regular mini-lattice.
    - Elsewhere, only a few solid square blocks from the enlarged version are already present.

    Rule:
    - Read the legend pixels as lattice coordinates, infer the block size and spacing from the existing square anchors, and vote for the unique legend-to-block offset.
    - Materialize every legend pixel as a same-colored square block at that inferred scale.
    - Do not keep the original legend in the answer; return only the bounding rectangle of the completed block field.

    Key insight: the top-left sprite is just a temporary blueprint, so once the scale and offset are known from the anchors, the true answer is the finished block arrangement cropped to its own extent.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    singles, blocks = _extract_components(g, bg)
    assert singles and blocks, "grid must contain legend pixels and main blocks"

    legend_pts = _extract_legend(singles, h, w)
    block_size, blocks = _dominant_blocks(blocks)
    step = _infer_step(blocks)
    assert step > 0, "block step must be positive"
    base_r, base_c = _infer_base(legend_pts, blocks, step)

    for rr, cc, _col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, (
            "legend block projection must fit in canvas"
        )

    return _render_cropped_blocks(
        legend_pts,
        bg=bg,
        block_size=block_size,
        step=step,
        base_r=base_r,
        base_c=base_c,
    )


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


def _pick_staircase() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, width in enumerate(widths) for c in range(width)]
    pairs = [(r, c) for r, width in enumerate(widths) for c in range(width - 1)]
    return pts, pairs


def _random_color_counts(total: int, k: int) -> list[int]:
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, count in enumerate(counts) if count < MAX_COUNT_PER_COLOR]
        idx = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[idx]))
        counts[idx] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(
    pts: list[tuple[int, int]],
    anchor_pair: tuple[int, int],
    palette: list[int],
) -> dict[tuple[int, int], int] | None:
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

    def allowed(pos: tuple[int, int], col: int) -> bool:
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

    def search() -> bool:
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
            colors[pos] = int(col)
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


def _render_full_canvas(
    *,
    size: int,
    bg: int,
    legend_colors: dict[tuple[int, int], int],
    pts: list[tuple[int, int]],
    base_r: int,
    base_c: int,
    step: int,
    block_size: int,
) -> np.ndarray:
    grid = np.full((size, size), int(bg), dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        grid[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = int(col)
    for rr, cc in pts:
        col = int(legend_colors[(rr, cc)])
        br = base_r + rr * step
        bc = base_c + cc * step
        grid[br : br + block_size, bc : bc + block_size] = col
    return grid


def _expected_output(
    *,
    bg: int,
    legend_colors: dict[tuple[int, int], int],
    base_r: int,
    base_c: int,
    step: int,
    block_size: int,
) -> np.ndarray:
    legend_pts = [(rr, cc, int(col)) for (rr, cc), col in legend_colors.items()]
    return _render_cropped_blocks(
        legend_pts,
        bg=int(bg),
        block_size=int(block_size),
        step=int(step),
        base_r=int(base_r),
        base_c=int(base_c),
    )


def _gen_attempt() -> np.ndarray | None:
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

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    full = _render_full_canvas(
        size=size,
        bg=bg,
        legend_colors=legend_colors,
        pts=pts,
        base_r=base_r,
        base_c=base_c,
        step=step,
        block_size=block_size,
    )
    out = _expected_output(
        bg=bg,
        legend_colors=legend_colors,
        base_r=base_r,
        base_c=base_c,
        step=step,
        block_size=block_size,
    )

    inp = full.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for rr, cc in pts:
        if (rr, cc) in keep:
            continue
        br = base_r + rr * step
        bc = base_c + cc * step
        inp[br : br + block_size, bc : bc + block_size] = bg

    # Anti-degeneration check:
    # If we merely tight-cropped the original full output without erasing legend,
    # the legend would still affect the bbox. The combo must differ from that.
    naive_crop = _tight_crop_non_bg(full, bg)
    if np.array_equal(naive_crop, out):
        return None

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
