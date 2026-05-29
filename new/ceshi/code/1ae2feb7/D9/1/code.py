# Variant of 1ae2feb7
# Dimension: D9 — 分隔条结构 (obstacle)
# Change: 单条垂直分隔列——一整列（允许缺1个格子）填充同色像素，将网格分为bar_side和ext_side... → 双分隔条（两条垂直线）

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D9 variant
# ==================================
# a) Concrete behavior
#    原题只有一条近乎贯穿全高的同色竖分隔条，所有水平色条都在分隔条的一侧，另一侧整块为空，
#    输出是在空白侧按“色条长度 = 打点步长”的周期延伸。
#    本变体保留“水平色条 + 周期打点 + 近处分组优先且不覆盖”的核心机制，只把单分隔条换成
#    两条同色竖分隔条。于是网格被切成 [左色条区 | 中间目标区 | 右色条区] 三段：
#    - 左侧色条只向右打到中间区；
#    - 右侧色条只向左打到中间区；
#    - 输入中的中间区完全为空，输出中的中间区同时承接两侧的周期点列。
#    具体小例子：若左分隔条右边第一行左侧有长度 3 的色条，则它在中间区的该行填
#    c = left_sep+1, left_sep+4, left_sep+7, ...；若同一行右侧也有长度 2 的色条，则它在
#    c = right_sep-1, right_sep-3, right_sep-5, ... 反向填色。若两侧都想填同一格，就沿用
#    “已有颜色不覆盖”，而全局处理顺序推广为“先按各自离分隔条的距离排序；同距离时左侧先于右侧”。
#    这样中间区会出现从两边向内的交错点列，而不是原题那种单向展开。
#
# b) Complexity calibration
#    若只是放两条分隔条但只让一侧有色条，任务会退化成原题。为避免这一点，生成器强制：
#    - 左右两侧在每个样本里都必须至少出现一次；
#    - 至少有一条活动行同时在左右两侧都有色条；
#    - 活动行仍然稀疏（相邻活动行之间至少隔 1 空行），避免中间区视觉过乱。
#    每侧每行仍只放 1-2 条相邻色条，不引入新的对象类别，复杂度增加主要来自双向竞争。
#
# c) Hidden assumptions in the original main()
#    - _find_sep_col 假设只有一条竖分隔列；双分隔条下必须改为寻找两列同色候选列。
#    - 原版 left_mass/right_mass 用来判断“哪边是 bars, 哪边是 ext”；现在两边都是 bars，
#      中间才是 ext，因此这套判定失效，必须显式把左右外侧都当成色条来源。
#    - 原版 bars 列表只收集一侧区间；现在要分别扫描 [0,left_sep) 和 (right_sep,w)。
#    - 原版打点循环默认目标区是整块单侧区域；现在左右两边都只能在 (left_sep,right_sep)
#      之间打点。
#    - 原版“近条优先”只在单侧排序；现在需要推广到双侧。这里保持原有思想，仍以“离本侧分隔条
#      的距离”作为首要优先级，只新增一个固定的左右并列 tie-break，避免出现不确定覆盖。
#
# d) Placement legality
#    生成器显式采样 left_width / mid_width / right_width，确保：
#    - 左右外侧都有足够宽度容纳 1-2 条色条和可选 1 格分隔条间隙；
#    - 中间区足够宽，能同时看见左右打点；
#    - 两条分隔条不会与色条重叠，也不会越界。
#    分隔条的可选缺口只放在非活动行上，避免破坏“某行色条紧邻分隔条”的可读性。
#
# e) Visual observability
#    解题者肉眼可见的线索是：
#    - 有两条同色竖线，而不是一条；
#    - 左右两侧都出现了原题风格的水平色条；
#    - 输出的新增像素只出现在两条竖线之间；
#    - 同一行里有时能同时看到从左右边界往中间推进的周期点列。
#    规则只用几何关系与长度，不依赖颜色 id、黑色排序等不可见因素。
#
# f) Visual cleanliness
#    中间区只承载单层打点，不画额外标记；活动行做稀疏采样，避免多行叠杂。左右外侧与中间区被
#    两条竖线天然分隔，因此人眼能清楚区分“发射源”和“目标区”。
#
# g) Color inference
#    颜色本身不发生映射，只是沿着本行按原色延伸，因此不需要推断复杂的颜色替换。调色板仍控制在
#    5-8 种非零颜色内，且分隔条颜色与色条颜色保持互斥。
#
# h) Generator-side non-degeneracy checks
#    output-first 构造完成后，除了验证 main(inp)==out，还要求：
#    - 左右两侧都存在活动行；
#    - 至少一条活动行在输出中同时保留了左右两侧的可见贡献；
#    这样不会长期退化回“只有一侧在起作用”的原题。
#
# i) New solver assumptions guaranteed by generator
#    main() 新假设：
#    - 恰有两条同色的近乎满高竖分隔条；
#    - 分隔条外侧才有输入色条，中间区输入为空；
#    - 水平色条不会跨越分隔条；
#    - 活动行稀疏，非分隔条列不会伪装成“近乎满高的同色列”。
#    generate_input() 会按这些约束构造样本。


def _find_sep_cols(g: np.ndarray, bg: int) -> tuple[int, int, int] | tuple[None, None, None]:
    h, w = g.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            candidates.append((c, int(vals[0])))
    best: tuple[int, int, int] | None = None
    for i, (c0, col0) in enumerate(candidates):
        for c1, col1 in candidates[i + 1 :]:
            if col0 != col1:
                continue
            if best is None or (c1 - c0) > (best[1] - best[0]):
                best = (c0, c1, col0)
    if best is None:
        return None, None, None
    return best


def _iter_runs_1d(row: np.ndarray, c_lo: int, c_hi: int, bg: int):
    c = c_lo
    while c < c_hi:
        if row[c] == bg:
            c += 1
            continue
        col = int(row[c])
        c0 = c
        while c < c_hi and row[c] == col:
            c += 1
        yield c0, c - 1, col


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    left_sep, right_sep, _sep_color = _find_sep_cols(g, bg)
    if left_sep is None or right_sep is None:
        return g.copy()

    out = g.copy()
    bars: list[tuple[int, int, int, int, int]] = []
    for r in range(h):
        for c0, c1, col in _iter_runs_1d(g[r, :], 0, left_sep, bg):
            length = c1 - c0 + 1
            dist = (left_sep - 1) - c1
            bars.append((dist, 0, r, length, col))
        for c0, c1, col in _iter_runs_1d(g[r, :], right_sep + 1, w, bg):
            length = c1 - c0 + 1
            dist = c0 - (right_sep + 1)
            bars.append((dist, 1, r, length, col))

    bars.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    for _dist, side, r, length, col in bars:
        if length <= 0:
            continue
        if side == 0:
            for cc in range(left_sep + 1, right_sep, length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(right_sep - 1, left_sep, -length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

SIDE_MIN, SIDE_MAX = 3, 5
MID_MIN, MID_MAX = 5, 10

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 3, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.15

MAX_ATTEMPTS = 400


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = side_len - sep_gap
    if capacity <= 0:
        raise ValueError("invalid side capacity")
    two_bars = capacity >= 2 and rng.random() < P_TWO_BARS

    if not two_bars:
        length = int(rng.integers(1, capacity + 1))
        return [(length, primary)], sep_gap

    near_len = int(rng.integers(1, capacity))
    far_len = int(rng.integers(1, capacity - near_len + 1))
    others = [c for c in palette if c != primary]
    colors = [primary, int(rng.choice(others))]
    if rng.random() < P_SWAP_COLORS:
        colors.reverse()
    return [(near_len, colors[0]), (far_len, colors[1])], sep_gap


def _place_side_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    is_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    if is_left:
        cursor = sep_col - sep_gap
        for length, color in bars:
            out[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            out[row, cursor : cursor + length] = color
            cursor += length


def _paint_middle_from_op(
    out: np.ndarray,
    row: int,
    left_sep: int,
    right_sep: int,
    side: int,
    length: int,
    color: int,
) -> int:
    hits = 0
    if side == 0:
        positions = range(left_sep + 1, right_sep, length)
    else:
        positions = range(right_sep - 1, left_sep, -length)
    for cc in positions:
        if out[row, cc] == 0:
            out[row, cc] = color
            hits += 1
    return hits


def _choose_primary(rng: np.random.Generator, palette: list[int], exclude: set[int]) -> int:
    choices = [c for c in palette if c not in exclude]
    if not choices:
        choices = palette
    return int(rng.choice(choices))


def _choose_row_states(rng: np.random.Generator, count: int) -> list[str]:
    states = ["B", "L", "R"]
    while len(states) < count:
        states.append(str(rng.choice(["L", "R", "B"])))
    rng.shuffle(states)
    return states


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    feasible: list[tuple[int, int, int]] = []
    for left_width in range(SIDE_MIN, SIDE_MAX + 1):
        for right_width in range(SIDE_MIN, SIDE_MAX + 1):
            mid_width = w - left_width - right_width - 2
            if MID_MIN <= mid_width <= MID_MAX:
                feasible.append((left_width, right_width, mid_width))
    if not feasible:
        return None

    left_width, right_width, _mid_width = feasible[int(rng.integers(0, len(feasible)))]
    left_sep = left_width
    right_sep = w - right_width - 1

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    if ACTIVE_MIN > max_active:
        return None
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_row_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_row_set]
    row_states = _choose_row_states(rng, active_count)

    out = np.zeros((h, w), dtype=int)
    out[:, left_sep] = sep_color
    out[:, right_sep] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, left_sep] = 0
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, right_sep] = 0

    ops: list[tuple[int, int, int, int, int]] = []
    side_active = {0: 0, 1: 0}
    both_rows: list[int] = []
    visible_both_rows = 0

    for row, state in zip(active_rows, row_states):
        per_row_ops: list[tuple[int, int, int, int, int]] = []
        used_row_colors: set[int] = set()

        if state in ("L", "B"):
            primary = _choose_primary(rng, palette, used_row_colors)
            used_row_colors.add(primary)
            bars, sep_gap = _sample_row_bars(rng, palette, primary, left_width)
            _place_side_bars(out, row, left_sep, True, bars, sep_gap)
            side_active[0] += 1
            cursor = left_sep - sep_gap
            for length, color in bars:
                dist = (left_sep - 1) - (cursor - 1)
                per_row_ops.append((dist, 0, row, length, color))
                cursor -= length

        if state in ("R", "B"):
            primary = _choose_primary(rng, palette, used_row_colors)
            used_row_colors.add(primary)
            bars, sep_gap = _sample_row_bars(rng, palette, primary, right_width)
            _place_side_bars(out, row, right_sep, False, bars, sep_gap)
            side_active[1] += 1
            cursor = right_sep + 1 + sep_gap
            for length, color in bars:
                dist = cursor - (right_sep + 1)
                per_row_ops.append((dist, 1, row, length, color))
                cursor += length

        if state == "B":
            both_rows.append(row)
        ops.extend(per_row_ops)

    ops.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4]))
    row_visible_hits: dict[int, dict[int, int]] = {row: {0: 0, 1: 0} for row in both_rows}
    for dist, side, row, length, color in ops:
        hits = _paint_middle_from_op(out, row, left_sep, right_sep, side, length, color)
        if row in row_visible_hits:
            row_visible_hits[row][side] += hits

    visible_both_rows = sum(1 for row in both_rows if row_visible_hits[row][0] > 0 and row_visible_hits[row][1] > 0)

    inp = out.copy()
    inp[:, left_sep + 1 : right_sep] = 0

    if side_active[0] == 0 or side_active[1] == 0:
        return None
    if not both_rows or visible_both_rows == 0:
        return None
    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
