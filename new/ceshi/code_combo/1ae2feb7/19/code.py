# Combo variant of 1ae2feb7 (index 19)
# combo_spec: D5/0+D8/2+D13/2
# Name: 擦除源后行号间隔颜色递变

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the fused variant
# =====================================
# a) 具体化与行为追踪
# - 仍然有一根几乎贯穿全高的单色竖直分隔条，把网格分成 bar_side 和 ext_side。
# - 输入里：bar_side 上若干活动行有水平纯色色条，ext_side 全空。
# - 输出里：先把每条色条向 ext_side 投射，但两条核心规则同时改变：
#   1) 步长不再是 bar 长度，而是该行的 1-based 行号 step=r+1。
#   2) 新涂上的第 n 个点不再固定用源条颜色，而是沿“输入中可见条色顺序”循环递变。
#      每条 bar 的第一个成功涂点仍从该 bar 自己的颜色开始，后续成功涂点依次取调色板的下一色。
# - 全部涂完后，再把 bar_side 上原始色条擦掉；输出最终只剩分隔条和 ext_side 的彩色点列。
# - 小例子（bars_on_left=True, sep_col=4，5 条唯一颜色按上到下是 [1,3,4,6,8]）：
#   某行 r=2（step=3）有一条颜色 4 的 bar。原题会按 bar_length 去涂同色点。
#   本变体会在列 5,8,11,... 尝试涂点，颜色依次是 4,6,8,1,...；若某格已被更近 bar 占用则跳过，
#   且只有成功涂上的点才推进颜色计数。最后把该行输入中的那条 4 擦成背景。
# - 因而与原题相比，差异体现在三个环节协同发生：
#   步长由“横向长度”改为“纵向行号”，颜色由“常量复制”改为“可见调色板循环”，
#   且源条在输出末尾被擦除，迫使解题者同时从输入中的条色顺序和输出中的密度变化去归纳规则。
#
# b) 复杂度校准
# - solver 仍支持同一行 1-2 条 run，但 generator 刻意主要生成“每个活动行 1 条 bar”。
#   原因是 D8 把步长改成按行号后，同一行不同 bar 的步长和相位完全一致；保留 D5 擦除后，
#   远条通常不会留下任何输出痕迹，只会徒增推断噪声。
# - 为了让 D13 的颜色递变可见，generator 固定生成 5 条互异条色，形成长度为 5 的可见循环；
#   这既足够展示规律，也不至于把颜色种类堆得太多。
# - 为了让 D8 的行号步长可见，active rows 会覆盖顶部和底部，并强制 ext_side 足够宽，
#   同时拒绝那些“行号步长版”和“长度步长版”输出过于接近的样本。
#
# c) 隐式假设审查
# - `_find_sep_col` 仍假设存在一列“非背景像素几乎全同色”的分隔条。融合后不变。
# - `left_mass/right_mass` 仍依赖输入时只有一侧有 bar；generator 保证 ext_side 清空。
# - `_iter_runs_1d` 仍要求 bar 是同色水平 run；颜色循环只影响输出，不影响输入编码。
# - 原题中的 `step = length` 假设被 D8 直接打破，必须替换为 `step = r + 1`。
# - 原题中的 `out[r, cc] = col` 假设“同一 bar 的所有点颜色恒定”；D13 打破这一点，
#   新逻辑必须维护“该 bar 已成功涂了多少个点”，并按循环取色。
# - D5 进一步打破了“out=g.copy() 后保留源条”的默认假设。必须先根据输入 bar 完成全部投射，
#   再统一擦除源条；若边擦边算，会破坏后续 bar 提取所需的信息。
# - 颜色循环依赖一个人类可见的调色板顺序。这里采用“按输入从上到下、每行从靠近分隔条到远离分隔条”
#   首次见到的条色顺序。这个顺序完全由可见结构决定，不依赖颜色 id 数值大小。
#
# d) 放置合法性
# - generator 保证所有 bar 都放在同一侧、在边界内、互不重叠，且与分隔条之间只允许 0 或 1 格空隙。
# - active rows 两两至少隔 1 空行，避免多行输出的彩色点列在视觉上连成一团。
# - ext_side 宽度被压到较长区间，使 step=r+1 仍能产生多个点；同时 bar_side 仍保留足够宽度容纳 bar。
# - 因为输出会擦除 bar_side，任何依赖源条“遗留在输出里”提供线索的布局都不被采用。
#
# e) 视觉可观察性
# - 人类能从训练对直接看到三件事：输入有横条、输出中横条消失、另一侧出现密度随行变化且颜色递变的点列。
# - 颜色循环不是隐式数值映射，而是直接由输入里 5 条不同颜色的可见顺序决定；
#   每一行的输出都从该行自身颜色起步，因此“起点色”和“循环顺序”都能从样例中对照出来。
# - 行号步长通过“顶行更密、底行更疏”的稳定现象表现出来；generator 强制出现顶部和底部活动行。
#
# f) 可视化整洁度
# - 输出只保留 separator + ext_side 点列，bar_side 被清空，因此视觉层次比原题还更干净。
# - 每行只放 1 条 bar 的主生成策略避免了同一行多个 latent bar 共享同一组 ext positions 所造成的冗余。
# - no-overwrite 规则保持不变，任何更远对象都不会在同一单元格产生混叠。
#
# g) 变化的可推断性
# - 颜色变化有明确的结构线索：调色板顺序来自输入里可见的条色顺序，而不是任意换色表。
# - 颜色种类固定为 5 条唯一条色，既足够形成明显循环，也不会让示例沦为纯记忆颜色排列。
# - generator 会量化检查：融合输出必须同时区别于
#   1) 擦除+颜色循环但仍用长度步长，
#   2) 擦除+行号步长但仍用固定条色，
#   否则样本判为退化并丢弃。


def _find_sep_col(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            return c, int(vals[0])
    return None, None


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


def _row_step(r: int) -> int:
    return int(r) + 1


def _bars_on_left(g: np.ndarray, sep_col: int, bg: int) -> bool:
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    return left_mass >= right_mass


def _extract_bars_and_cycle(
    g: np.ndarray, sep_col: int, bars_on_left: bool, bg: int
) -> tuple[list[tuple[int, int, int, int, int, int]], list[int]]:
    h, w = g.shape
    bars: list[tuple[int, int, int, int, int, int]] = []
    cycle: list[int] = []
    seen: set[int] = set()

    for r in range(h):
        runs = list(
            _iter_runs_1d(
                g[r, :],
                0 if bars_on_left else sep_col + 1,
                sep_col if bars_on_left else w,
                bg=bg,
            )
        )
        ordered_runs = list(reversed(runs)) if bars_on_left else runs
        for c0, c1, col in ordered_runs:
            length = c1 - c0 + 1
            dist = (sep_col - 1) - c1 if bars_on_left else c0 - (sep_col + 1)
            bars.append((dist, r, c0, c1, length, col))
            if col not in seen:
                seen.add(col)
                cycle.append(col)

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    return bars, cycle


def _solve_core(
    grid: np.ndarray,
    *,
    use_row_step: bool,
    use_cycle: bool,
    erase_source: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()
    bars_on_left = _bars_on_left(g, sep_col, bg)
    bars, cycle = _extract_bars_and_cycle(g, sep_col, bars_on_left, bg)
    cycle_index = {col: idx for idx, col in enumerate(cycle)}

    for _, r, _c0, _c1, length, col in bars:
        step = _row_step(r) if use_row_step else length
        if step <= 0:
            continue

        if bars_on_left:
            positions = range(sep_col + 1, g.shape[1], step)
        else:
            positions = range(sep_col - 1, -1, -step)

        painted = 0
        for cc in positions:
            if int(out[r, cc]) != bg:
                continue
            if use_cycle and cycle:
                idx = (cycle_index[col] + painted) % len(cycle)
                out[r, cc] = cycle[idx]
            else:
                out[r, cc] = col
            painted += 1

    if erase_source:
        for _, r, c0, c1, _length, _col in bars:
            out[r, c0 : c1 + 1] = bg

    return out


# Overview:
#   Project each source bar across the separator using a row-dependent stride, cycle the
#   paint color through the visible bar palette, then erase the original source bars.
# Input structure:
#   One nearly solid vertical separator column splits the grid into a populated bar side
#   and an empty extension side. Bars are horizontal monochrome runs on a few spaced rows.
# Rule:
#   Read visible bar colors in input order (top-to-bottom, and within a row from near the
#   separator outward) to form a palette cycle. Each bar paints the extension side at stride
#   `row_index + 1`; its first successful dot uses its own color, later successful dots use
#   successive colors in that cycle. Existing non-background cells are never overwritten.
#   After all painting, erase the original source bars to background.
# Key insight:
#   The output keeps only the separator and the transformed dots, so the solver must combine
#   two visible cues from the input-output pairs: dot spacing comes from row position, while
#   dot colors advance through the input-visible palette rather than staying constant.
def main(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, use_row_step=True, use_cycle=True, erase_source=True)


# --------------- generator ---------------

H_MIN, H_MAX = 9, 13
W_MIN, W_MAX = 16, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 4, 9
EXT_SIDE_MIN, EXT_SIDE_MAX = 8, 14

ACTIVE_COUNT = 5
TOTAL_BARS = ACTIVE_COUNT

P_SEP_GAP = 0.15
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 500
MIN_DIFF_VS_LENGTH_STEP = 3
MIN_DIFF_VS_CONSTANT_COLOR = 4
MIN_ROWS_WITH_MULTI_COLORS = 3
MIN_ROWS_WITH_REPEAT = 3

# Alternate the populated side across accepted calls so self-tests reliably cover both mirror layouts.
_GEN_CALL_COUNT = 0


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    palette = list(map(int, rng.choice(bar_pool, size=TOTAL_BARS, replace=False)))
    return sep_color, palette


def _choose_active_rows(rng: np.random.Generator, h: int) -> list[int]:
    top_band = list(range(min(3, h)))
    bottom_band = list(range(max(0, h - 3), h))

    for _ in range(300):
        chosen = {int(rng.choice(top_band)), int(rng.choice(bottom_band))}
        candidates = [r for r in range(h) if r not in chosen]
        rng.shuffle(candidates)
        for r in candidates:
            if all(abs(r - prev) >= 2 for prev in chosen):
                chosen.add(int(r))
            if len(chosen) == ACTIVE_COUNT:
                rows = sorted(chosen)
                if rows[0] <= 2 and rows[-1] >= h - 3:
                    return rows
                break
    raise RuntimeError("failed to choose informative active rows")


def _sample_row_bar(
    rng: np.random.Generator,
    primary: int,
    bar_side_len: int,
    row: int,
    ext_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise RuntimeError("invalid bar capacity")

    step = _row_step(row)
    short_cap = min(4, capacity)
    preferred = [
        length
        for length in range(1, short_cap + 1)
        if length != step and max(step, length) <= ext_side_len
    ]
    if preferred:
        length = int(rng.choice(preferred))
    else:
        informative = [length for length in range(1, capacity + 1) if length != step]
        choices = informative if informative else list(range(1, capacity + 1))
        length = int(rng.choice(choices))
    return [(length, primary)], sep_gap


def _place_bars(
    grid: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, color in bars:
            grid[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            grid[row, cursor : cursor + length] = color
            cursor += length


def _variant_metrics(inp: np.ndarray) -> tuple[int, int, int, int]:
    fused = _solve_core(inp, use_row_step=True, use_cycle=True, erase_source=True)
    length_step = _solve_core(inp, use_row_step=False, use_cycle=True, erase_source=True)
    constant_color = _solve_core(inp, use_row_step=True, use_cycle=False, erase_source=True)

    diff_vs_length = int(np.sum(fused != length_step))
    diff_vs_constant = int(np.sum(fused != constant_color))

    bg = get_bg(inp)
    sep_col, _ = _find_sep_col(inp, bg)
    if sep_col is None:
        return diff_vs_length, diff_vs_constant, 0, 0

    bars_on_left = _bars_on_left(inp, sep_col, bg)
    rows_with_multi_colors = 0
    rows_with_repeat = 0
    for r in range(inp.shape[0]):
        ext = fused[r, sep_col + 1 :] if bars_on_left else fused[r, :sep_col]
        painted = ext[ext != bg]
        if painted.size >= 2:
            rows_with_repeat += 1
        if len(set(map(int, painted))) >= 2:
            rows_with_multi_colors += 1

    return diff_vs_length, diff_vs_constant, rows_with_multi_colors, rows_with_repeat


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))
    ext_side_len = w - 1 - bar_side_len

    if ACTIVE_COUNT > (h + 1) // 2:
        return None

    global _GEN_CALL_COUNT
    bars_on_left = (_GEN_CALL_COUNT % 2 == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)
    active_rows = _choose_active_rows(rng, h)
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        inp[gap_row, sep_col] = 0

    for row, color in zip(active_rows, palette):
        bars, sep_gap = _sample_row_bar(rng, color, bar_side_len, row, ext_side_len)
        _place_bars(inp, row, sep_col, bars_on_left, bars, sep_gap)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None

    diff_vs_length, diff_vs_constant, rows_with_multi, rows_with_repeat = _variant_metrics(inp)
    if diff_vs_length < MIN_DIFF_VS_LENGTH_STEP:
        return None
    if diff_vs_constant < MIN_DIFF_VS_CONSTANT_COLOR:
        return None
    if rows_with_multi < MIN_ROWS_WITH_MULTI_COLORS:
        return None
    if rows_with_repeat < MIN_ROWS_WITH_REPEAT:
        return None

    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
