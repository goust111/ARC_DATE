# Variant of 1ae2feb7
# Dimension: D14 — 涂色粒度（每次涂色面积） (trajectory)
# Change: 每个周期位置涂1个像素——涂点是单个格子... → 涂色扩展为十字形

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D14 variant
# -----------------------------------
# source_task: None
#
# a) 具体化与行为追踪
# - 原题中，条形对象位于分隔列一侧；每个条形按“长度 = 周期”在另一侧同一行单点落色。
# - 本变体保留“分隔列识别 / 条形距离排序 / 周期步长 / 不覆盖已有颜色”这几个核心规则，
#   唯一改动是单次落色的面积：从 (r, c) 一个像素，扩成同列的 (r-1, c), (r, c), (r+1, c)。
# - 因此，输出里每个周期位置不再是孤立点，而是 2-3 格高的短竖线；当两个活跃行相距 2 时，
#   它们会在中间空行上发生挡色交互。
# - 小例子：若分隔列右侧为空，某行 r 上有长度 3 的条，原题会在 c = s+1, s+4, s+7... 处仅给
#   out[r, c] 上色；本变体改为尝试给 out[r-1, c], out[r, c], out[r+1, c] 上色，且每个格子
#   单独遵守“非背景则不覆盖”。
#
# b) 复杂度校准
# - 只引入一种新的落色形状（固定的纵向 3 格），不引入额外颜色规则、旋转规则或多档模式。
# - 为了避免画面过密，生成器将活跃行数上限略收紧到 4；但仍保留 1-2 条条形、左右两侧、与分隔列
#   间偶发 1 格空隙等原题自由度。
#
# c) 原 solver 中与变体相关的隐式前提审查
# - 原 `for cc ... if out[r, cc] == bg: out[r, cc] = col` 默认“一次只写本行一个格子”。
#   变体下该前提失效，必须改成逐个检查 r-1/r/r+1 三个格子，不能整列无条件赋值。
# - 原 generator 的 `_paint_extension` 是逐行独立模拟；在变体下不同条会跨行互相挡色，
#   因此生成器必须改成和 solver 一致的“全局收集条形后统一排序再落色”。
# - 原排序 `bars.sort(key=(dist, row, c0))` 仍保留。这样“最近条优先”不变，只是其影响范围从
#   本行扩大到相邻行。
#
# d) 放置合法性
# - 输入中的条形仍只放在分隔列一侧，所以不会与扩展区重叠。
# - 十字形落色只发生在扩展侧，且对每个候选格单独做边界检查与背景检查，因此不会越界，也不会覆盖
#   已存在的颜色。
# - 生成器仍采用 output-first：先放条形，再按 solver 同步的顺序模拟十字形扩展，最后清空扩展侧
#   得到输入，并用 `main(input) == output` 验证。
#
# e) 视觉可观察性
# - 这个变体的可见特征是：输出里出现大量纵向 2-3 格短线，而不是原题的单行点列。
# - 生成器额外拒绝“新旧逻辑输出完全相同”的样本，并要求至少出现一个只会由跨行扩展产生的格子，
#   从而避免退化回原题。
#
# f) 可视化整洁度
# - 条形侧仍与原题相同，视觉锚点稳定；扩展侧虽然更密，但只增加一层简单的纵向扩张。
# - 活跃行仍保持至少隔 1 个空行，因此原始条形彼此清楚；允许扩展结果在空行上交互，但整体仍可读。
#
# g) 颜色变化可推断性
# - 颜色不做映射，只沿用各条形自身颜色；可推断线索完全来自“哪个条决定了哪个周期列”以及
#   “十字形会往上下多染 1 格”。
#
# 额外审查：新 `main()` 的隐式假设
# - 仍假设输入扩展侧为空，否则 `_find_sep_col` 与 bar-side 质量判断可能失真；生成器显式保证这点。
# - 仍假设所有条形都位于分隔列同一侧；生成器沿用原布局保证这点。
# - 现在假设输出模拟顺序与 solver 完全一致，否则跨行挡色会不匹配；生成器通过同一排序键保证。


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


def _paint_vertical_mark(out: np.ndarray, row: int, col: int, color: int, bg: int) -> None:
    h, _w = out.shape
    for rr in (row - 1, row, row + 1):
        if 0 <= rr < h and int(out[rr, col]) == bg:
            out[rr, col] = color


def _collect_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> list[tuple[int, int, int, int, int, int]]:
    bars: list[tuple[int, int, int, int, int, int]] = []
    h, w = g.shape
    for r in range(h):
        if bars_on_left:
            runs = _iter_runs_1d(g[r, :], 0, sep_col, bg=bg)
            for c0, c1, col in runs:
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, c1, length, col))
        else:
            runs = _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg)
            for c0, c1, col in runs:
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, c1, length, col))
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    return bars


def _apply_extension(
    out: np.ndarray,
    bars: list[tuple[int, int, int, int, int, int]],
    sep_col: int,
    bars_on_left: bool,
    bg: int,
    *,
    cross: bool,
) -> None:
    _h, w = out.shape
    for _dist, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if cross:
                _paint_vertical_mark(out, r, cc, col, bg)
            elif int(out[r, cc]) == bg:
                out[r, cc] = col


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    bars = _collect_bars(g, sep_col, bars_on_left, bg)
    _apply_extension(out, bars, sep_col, bars_on_left, bg, cross=True)
    return out


# --------------- generator (D14 variant) ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 4

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5

P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    """Pick `count` row indices in [0, h) with pairwise spacing >= 2."""
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
    bar_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    """Return (bars, sep_gap) where bars is a list of (length, color) ordered near->far."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    two_bars = rng.random() < P_TWO_BARS

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


def _place_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> list[tuple[int, int, int, int, int, int]]:
    """Paint bars and return solver-compatible bar records."""
    records: list[tuple[int, int, int, int, int, int]] = []
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, color in bars:
            c1 = cursor - 1
            c0 = cursor - length
            out[row, c0:cursor] = color
            dist = (sep_col - 1) - c1
            records.append((dist, row, c0, c1, length, color))
            cursor = c0
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            c0 = cursor
            c1 = cursor + length - 1
            out[row, cursor : cursor + length] = color
            dist = c0 - (sep_col + 1)
            records.append((dist, row, c0, c1, length, color))
            cursor = c1 + 1
    return records


def _variant_gain_metrics(
    inp: np.ndarray,
    out: np.ndarray,
    bars: list[tuple[int, int, int, int, int, int]],
    sep_col: int,
    bars_on_left: bool,
    active_rows: list[int],
) -> tuple[int, int]:
    baseline = inp.copy()
    _apply_extension(baseline, bars, sep_col, bars_on_left, bg=0, cross=False)
    delta = out != baseline
    total_gain = int(np.count_nonzero(delta))
    active_mask = np.zeros(out.shape[0], dtype=bool)
    active_mask[active_rows] = True
    cross_row_gain = int(np.count_nonzero(delta & (~active_mask[:, None])))
    return total_gain, cross_row_gain


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    bar_records: list[tuple[int, int, int, int, int, int]] = []
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        bar_records.extend(_place_bars(out, row, sep_col, bars_on_left, bars, sep_gap))

    bar_records.sort(key=lambda t: (t[0], t[1], t[2]))
    _apply_extension(out, bar_records, sep_col, bars_on_left, bg=0, cross=True)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), out):
        return None

    total_gain, cross_row_gain = _variant_gain_metrics(
        inp, out, bar_records, sep_col, bars_on_left, active_rows
    )
    if total_gain <= 0 or cross_row_gain <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
