# Variant of c4d067a0
# Dimension: D1 — 匹配条件：坐标映射方式 (match_cond)
# Change: 图例像素位置通过统一缩放因子+偏移量映射到网格位置，缩放因子由已有色块的间距与图例像素间距的比值确定... → 旋转后缩放映射

from __future__ import annotations

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


"""
design_analysis:
  source_task_baseline:
    reference: fc7cae8d
    takeaway: 参考题通过枚举少量离散几何变换来匹配控制线索，因此本变体也保持同等复杂度，只枚举 3 个旋转档位，而不引入更重的搜索。
  a_concretization_and_trace:
    input_view:
      - 左上角仍是 step=2 的单像素楼梯状图例，颜色和原题一致，不旋转。
      - 主区域仍只有两个已知 NxN 色块，作为锚点。
      - 不同点在于：主区域的块阵列对应的是“旋转后的图例格点坐标”，而不是原坐标。
    trace_example:
      - 设原图例某相邻锚点在格点坐标中是 (1,0,色A) 和 (1,1,色B)。
      - 原题会直接映射到 (base_r+1*step, base_c+0*step) 与 (base_r+1*step, base_c+1*step)。
      - 本变体若旋转 90 度，则它们先变成 (W-1,1) 与 (W-2,1)，再映射为同一列的上下两个块。
      - 差异点是：旧 solver 只会按未旋转坐标投票，遇到竖直锚点会得到错误 base；新 solver 必须先枚举旋转，再做同样的缩放/平移投票。
  b_complexity_calibration:
    - 只允许 90/180/270 三档旋转，控制自由度，保证 2-4 个样例里能看出规律。
    - 仍保留 2 个锚点，不增加对象种类；复杂度来自“方向变化”，不是额外噪声。
    - 图例继续使用不规则楼梯形，避免 180 度旋转时纯几何对称导致变化不可见。
  c_implicit_assumption_audit:
    - 原代码 `legend_pts -> base + (rr,cc)*step` 假设图例坐标与块坐标轴平行；变体中这条假设失效，必须替换成 `rotate(rr,cc) -> base + rotated*step`。
    - 原代码用颜色投票只求 `(base_r,base_c)`；变体中候选必须扩展为 `(rotation, base_r, base_c)`。
    - 原代码在绘制阶段直接按未旋转 legend_pts 填块；变体中若保留这写法，会把正确的输入还原成原题输出，属于静默错误。
    - 原代码对步长的 gcd 推断仍然成立，因为旋转不改变块阵列的轴对齐和相邻间距。
  d_placement_legality:
    - 生成器先选旋转后的格点集合，再按其最大行列坐标计算合法 base 范围，保证所有块不越界。
    - `base_c >= 8` 保留，避免块区覆盖左上角图例窗口。
    - 只保留旋转后对应的两个锚点块，其余块擦回背景，输入仍可被 solver 解析为“图例单像素 + 主区色块”两类对象。
  e_visual_observability:
    - 生成器拒绝“旋转后彩图例与原彩图例完全相同”的样本，避免退化成原题。
    - 90/270 度样本会把水平锚点变成竖直块对；180 度样本会把块对左右次序反转，三种情况都能直接观察。
  f_visual_cleanliness:
    - 图例始终局限在左上 12x12 小窗内，块区从更右侧开始，区域职责清晰。
    - 主区对象仍全是纯色方块，没有叠层，也不新增装饰标记。
  g_color_inference:
    - 不做颜色变换，只复用原图例颜色。
    - 锚点两色强制不同，并禁止同一有序横向色对在图例里重复出现，从而保证旋转+平移投票唯一。
  new_solver_assumptions_checked_against_generator:
    - solver 假设输入至少有两个同尺寸主区色块；generator 始终保留恰好两个。
    - solver 假设正确解在 90/180/270 中唯一；generator 用 `main(inp) == out` 反验并重试，保证唯一性。
    - solver 假设图例格点步长可由单像素坐标 gcd 推出；generator 固定为 step=2 且至少 3x2 楼梯形。
"""


def _rotate_coord(rr: int, cc: int, height: int, width: int, k: int) -> tuple[int, int]:
    """Rotate a lattice coord inside a height x width bounding box by k*90° CCW."""
    k %= 4
    if k == 0:
        return int(rr), int(cc)
    if k == 1:
        return int(width - 1 - cc), int(rr)
    if k == 2:
        return int(height - 1 - rr), int(width - 1 - cc)
    return int(cc), int(height - 1 - rr)


def _rotate_legend_points(
    legend_pts: list[tuple[int, int, int]], k: int
) -> list[tuple[int, int, int]]:
    height = max(rr for rr, _, _ in legend_pts) + 1
    width = max(cc for _, cc, _ in legend_pts) + 1
    rotated = []
    for rr, cc, col in legend_pts:
        nr, nc = _rotate_coord(rr, cc, height, width, k)
        rotated.append((int(nr), int(nc), int(col)))
    return rotated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a block layout whose legend must be rotated before projection.

    Input structure:
    - bg = most common color.
    - Top-left corner: a sparse single-pixel legend on a regular lattice, still shown in its original orientation.
    - Main area: two or more uniformly colored square blocks on a coarser lattice, representing a rotated copy of the legend.

    Rule:
    - Read the legend as lattice coordinates, then test 90/180/270-degree rotations of that coordinate pattern.
    - Use the existing blocks to identify which rotation and which scale/offset align the rotated legend with the block lattice.
    - After the correct rotation is found, place every missing NxN block using the same colors as the legend pixels.

    Key insight: The legend is no longer projected directly; the visible blocks are evidence of a rotated-then-scaled coordinate map.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
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
    diffs = drs + dcs
    assert diffs, "legend lattice must expose a measurable step"
    stepL = reduce(gcd, diffs)
    assert stepL > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))

    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert blocks, "must contain at least one block of the dominant block size"

    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    assert step_candidates, "existing blocks must expose a block-lattice step"
    step = reduce(gcd, step_candidates)
    assert step > 0, "block step must be positive"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, _s, col in blocks:
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))

    votes: dict[tuple[int, int, int], int] = {}
    rotated_by_k: dict[int, list[tuple[int, int, int]]] = {}
    for k in (1, 2, 3):
        rotated_pts = _rotate_legend_points(legend_pts, k)
        rotated_by_k[k] = rotated_pts
        for rr, cc, col in rotated_pts:
            if col not in by_color:
                continue
            for br0, bc0 in by_color[col]:
                key = (int(k), int(br0 - rr * step), int(bc0 - cc * step))
                votes[key] = votes.get(key, 0) + 1

    assert votes, "existing blocks must determine the rotated legend projection"
    max_votes = max(votes.values())
    best = [key for key, count in votes.items() if count == max_votes]
    assert len(best) == 1, "rotation and base offset must be uniquely determined"
    rot_k, base_r, base_c = best[0]

    for rr, cc, col in rotated_by_k[rot_k]:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, (
            "rotated legend block projection must fit in canvas"
        )
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out


# --------------- generator (rotation variant) ---------------

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
ROTATIONS = (1, 2, 3)


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
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE if STEP_MIN <= n + g <= STEP_MAX
]
_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, width in enumerate(widths) for c in range(width)]
    pairs = [(r, c) for r, width in enumerate(widths) for c in range(width - 1)]
    return pts, pairs


def _random_color_counts(total, k):
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, value in enumerate(counts) if value < MAX_COUNT_PER_COLOR]
        idx = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[idx]))
        counts[idx] += add
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


def _rotate_color_map(colors: dict[tuple[int, int], int], k: int) -> dict[tuple[int, int], int]:
    height = max(r for r, _ in colors) + 1
    width = max(c for _, c in colors) + 1
    rotated: dict[tuple[int, int], int] = {}
    for (rr, cc), col in colors.items():
        nr, nc = _rotate_coord(rr, cc, height, width, k)
        rotated[(int(nr), int(nc))] = int(col)
    return rotated


def _color_signature(colors: dict[tuple[int, int], int]) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted((int(r), int(c), int(col)) for (r, c), col in colors.items()))


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)
    rot_k = random.choice(ROTATIONS)

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k_colors = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k_colors)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    rotated_colors = _rotate_color_map(legend_colors, rot_k)
    if _color_signature(rotated_colors) == _color_signature(legend_colors):
        return None

    max_r = max(r for r, _ in rotated_colors)
    max_c = max(c for _, c in rotated_colors)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
    for (rr, cc), col in rotated_colors.items():
        br, bc = base_r + rr * step, base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = col

    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    legend_h = max(r for r, _ in legend_colors) + 1
    legend_w = max(c for _, c in legend_colors) + 1
    keep = {
        _rotate_coord(first[0], first[1], legend_h, legend_w, rot_k),
        _rotate_coord(second[0], second[1], legend_h, legend_w, rot_k),
    }

    inp = out.copy()
    for rr, cc in rotated_colors:
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
    raise RuntimeError("failed to generate valid c4d067a0 rotation input")
