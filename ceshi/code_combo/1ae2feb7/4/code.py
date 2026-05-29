# Combo variant of 1ae2feb7 (index 4)
# combo_spec: D2/0+D5/0
# Name: 连续填充后擦除源

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the combo variant
# =====================================
# a) 具体化与行为追踪
# - 输入仍是原题结构：一根几乎贯穿全高的竖直分隔列，把网格分成 source-bar 一侧和 ext 一侧；
#   2-5 个活动行在 source-bar 一侧放 1-2 条水平纯色色条，ext_side 在输入中全空。
# - 融合后的输出分两步：
#   1. 先按“近条优先、不可覆盖”把每条 bar 投影到 ext_side，但轨迹不再是单点，而是：
#      以 `step = bar_length` 为周期，从紧邻分隔列的 ext 端起，每隔 `step` 放一段
#      长度 `trail_len = floor(step / 2)` 的连续色段。
#   2. 全部 ext_side 投影完成后，把原来 bar_side 上的源条整段擦成背景色。
# - 小例子（bars_on_left=True, sep_col=4）：
#   输入行为 `[2,2,3,0,9,0,0,0,0,0,0,0]`，其中 `3` 是近条、长度 1；`2,2` 是远条、长度 2。
#   再换一个更能体现连续填充的例子：若近条长度 4、颜色 3，远条长度 2、颜色 2，
#   原题 ext 轨迹是 `3 . . . 3 . . . 3 ...`；本变体改成 `3 3 . . 3 3 . . 3 3 ...`。
#   然后 bar_side 的 `[2,2,3,3,3,3]` 会被擦掉，只剩 `0,0,0,0,9,3,3,0,0,3,3,...`。
# - 与单独 D2 相比，这里最终输出不保留源条；与单独 D5 相比，这里 ext_side 不是稀疏点，而是短条纹。
#
# b) 复杂度校准
# - 若允许长度 1 的 bar，则 `floor(1/2)=0`，擦除源条后该 bar 会在输出中完全消失，不可观察。
# - 若大量 bar 长度只有 2/3，则连续段长度都是 1，视觉上会退化成原题点阵或接近点阵。
# - 因此生成器一方面禁止长度 1 的 bar，并要求每条 bar 都在 ext_side 留下至少一个新像素；
#   另一方面强制每个样本至少有一条长度 >= 4 的 bar，并把 ext_side 下界抬到 5，确保 D2 差异可见。
# - 其余自由度仍保留原题风格：左右两侧随机、活动行 2-5、单条或双条、与 separator 允许 1 格 gap。
#
# c) 隐式假设审查
# - `_find_sep_col` 依赖“存在一列 >=H-1 个非背景像素且同色”。组合变体不改分隔列，成立。
# - `left_mass/right_mass` 仍基于输入判断哪侧是 bar_side；因为输入里 ext_side 仍被清空，成立。
# - `_iter_runs_1d` 把同色相邻像素视为一条 bar；生成器继续保证同一行双 bar 颜色不同，避免合并歧义。
# - `bars.sort(key=(dist,row,col))` 仍表达“近条优先”；连续段与擦除都不改变这一处理顺序。
# - 新风险 1：若先擦源条再继续读取 bar，会丢失长度/颜色信息，所以 solver 必须先收集 bars，再绘制，再擦除。
# - 新风险 2：连续段是切片写入，不是单点写入；必须按 segment 内逐像素 mask `== bg`，否则会覆盖近条结果。
# - 新风险 3：`trail_len = length // 2` 可能为 0；solver 安全跳过，generator 通过禁止长度 1 bar 来避免不可见源条。
#
# d) 放置合法性
# - 生成器仍按 output-first：先构造 latent 完整输出态（含源条和连续轨迹），再派生 input 和 final output。
# - 每行双 bar 的总长度不超过 bar_side 宽度；sep_gap 仅 0 或 1；因此不会越界，也不会穿过 separator。
# - ext_side 涂色只在本行进行，行间不相撞；同一行冲突仅来自多条 bar 竞争，仍由“近条优先 + 不覆盖”处理。
#
# e) 视觉可观察性
# - 解题者能同时看到两个稳定现象：输出的 bar_side 消失了；ext_side 变成周期性的短色段而非单点。
# - 生成器过滤掉与“擦除版原题”完全相同的样本，也过滤掉与“仅连续填充、不擦除”相比差异不足的退化情况。
# - 同时过滤任何“某条源 bar 在最终输出中没有留下可见痕迹”的样本，避免隐式、不可归纳的隐藏对象。
#
# f) 可视化整洁度
# - 擦除源条后，输出只剩 separator 与 ext_side 条纹，结构比原题更干净，不会同时堆叠“源条 + 投影”。
# - 连续段仍按活动行分散排布，活动行之间至少隔 1 个空行，避免多个条纹区粘连成块。
#
# g) 变化的可推断性
# - 没有颜色映射。颜色仍直接继承自源条；人类只需从 ext_side 段长/周期反推原始 bar 长度与颜色。
# - 每行只出现 1-2 种条色，颜色负担低；差异主要依靠位置、段长和“源条被擦除”这两个结构线索。
#
# new_solver_assumptions_checked_against_generator:
# - solver 假设输入 ext_side 全空，generator 统一清空该侧保证成立。
# - solver 假设至少存在一个长度 >= 1 的 bar；generator 每个活动行必放 1-2 条 bar。
# - solver 对 D2 可见性没有硬性要求，但 generator 通过 `require_long=True`、最短长度限制和可见性校验保证样本不退化。


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
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, c1, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, c1, length, col))
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    return bars


def _paint_periodic_trails(
    out: np.ndarray,
    bars: list[tuple[int, int, int, int, int, int]],
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> None:
    w = out.shape[1]
    for _dist, r, _c0, _c1, length, col in bars:
        step = int(length)
        trail_len = step // 2
        if step <= 0 or trail_len <= 0:
            continue
        if bars_on_left:
            for start in range(sep_col + 1, w, step):
                end = min(start + trail_len, w)
                seg = out[r, start:end]
                mask = seg == bg
                seg[mask] = col
        else:
            for start in range(sep_col - 1, -1, -step):
                lo = max(0, start - trail_len + 1)
                seg = out[r, lo : start + 1]
                mask = seg == bg
                seg[mask] = col


def _paint_periodic_points(
    out: np.ndarray,
    bars: list[tuple[int, int, int, int, int, int]],
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> None:
    w = out.shape[1]
    for _dist, r, _c0, _c1, length, col in bars:
        step = int(length)
        if step <= 0:
            continue
        if bars_on_left:
            for cc in range(sep_col + 1, w, step):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -step):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col


def _erase_source_bars(
    out: np.ndarray,
    bars: list[tuple[int, int, int, int, int, int]],
    bg: int,
) -> None:
    for _dist, r, c0, c1, _length, _col in bars:
        out[r, c0 : c1 + 1] = bg


def _solve_variant(grid: np.ndarray, painter, erase_source: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    bars_on_left = left_mass >= right_mass

    bars = _collect_bars(g, sep_col, bars_on_left, bg)
    out = g.copy()
    painter(out, bars, sep_col, bars_on_left, bg)
    if erase_source:
        _erase_source_bars(out, bars, bg)
    return out


# Overview:
#   Extend each source bar across the separator as periodic continuous segments, then erase the source bars.
# Input structure:
#   A near-full-height vertical separator splits the grid into a source-bar side and an empty extension side.
#   On 2-5 spaced rows, the source side contains 1-2 horizontal monochrome bars packed near the separator.
# Rule:
#   Process bars from nearest to farthest from the separator. For a bar of length L, paint on the opposite side
#   every L cells starting adjacent to the separator, but each hit is a continuous segment of length floor(L/2).
#   Do not overwrite non-background cells. After all segments are painted, erase the original source bars to background.
# Key insight:
#   The visible output keeps only the separator plus periodic short stripes, so the missing source bars must be
#   reconstructed from each row's stripe period, stripe length, and nearest-first occlusion pattern.
def main(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(grid, _paint_periodic_trails, erase_source=True)


# --------------- generator (combo: continuous-trail + erase-source) ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 4, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 5, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 500
MIN_BAR_LENGTH = 2


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
    bar_side_len: int,
    require_long: bool,
) -> tuple[list[tuple[int, int]], int] | None:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    if require_long and bar_side_len <= 4:
        sep_gap = 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        return None
    if require_long and capacity < 4:
        return None
    if not require_long and capacity < MIN_BAR_LENGTH:
        return None

    if require_long:
        if capacity == 4 or rng.random() < 0.65:
            length = int(rng.integers(4, capacity + 1))
            return [(length, primary)], sep_gap
        others = [c for c in palette if c != primary]
        long_near = bool(rng.integers(0, 2) == 0)
        long_len = int(rng.integers(4, capacity))
        rem = capacity - long_len
        if rem < MIN_BAR_LENGTH:
            return [(long_len, primary)], sep_gap
        other_len = int(rng.integers(MIN_BAR_LENGTH, rem + 1))
        other_color = int(rng.choice(others))
        if long_near:
            return [(long_len, primary), (other_len, other_color)], sep_gap
        return [(other_len, other_color), (long_len, primary)], sep_gap

    two_bars = (capacity >= 2 * MIN_BAR_LENGTH) and (rng.random() < P_TWO_BARS)
    if not two_bars:
        length = int(rng.integers(MIN_BAR_LENGTH, capacity + 1))
        return [(length, primary)], sep_gap

    near_len = int(rng.integers(MIN_BAR_LENGTH, capacity - MIN_BAR_LENGTH + 1))
    far_len = int(rng.integers(MIN_BAR_LENGTH, capacity - near_len + 1))
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
) -> None:
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, color in bars:
            out[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            out[row, cursor : cursor + length] = color
            cursor += length


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    w = out.shape[1]
    for length, color in bars:
        step = int(length)
        trail_len = step // 2
        if step <= 0 or trail_len <= 0:
            continue
        if bars_on_left:
            for start in range(sep_col + 1, w, step):
                end = min(start + trail_len, w)
                seg = out[row, start:end]
                mask = seg == 0
                seg[mask] = color
        else:
            for start in range(sep_col - 1, -1, -step):
                lo = max(0, start - trail_len + 1)
                seg = out[row, lo : start + 1]
                mask = seg == 0
                seg[mask] = color


def _bar_visibility_counts(
    width: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> list[int]:
    row = np.zeros(width, dtype=int)
    counts: list[int] = []
    for length, color in bars:
        before = int(np.count_nonzero(row))
        step = int(length)
        trail_len = step // 2
        if step > 0 and trail_len > 0:
            if bars_on_left:
                for start in range(sep_col + 1, width, step):
                    end = min(start + trail_len, width)
                    seg = row[start:end]
                    mask = seg == 0
                    seg[mask] = color
            else:
                for start in range(sep_col - 1, -1, -step):
                    lo = max(0, start - trail_len + 1)
                    seg = row[lo : start + 1]
                    mask = seg == 0
                    seg[mask] = color
        after = int(np.count_nonzero(row))
        counts.append(after - before)
    return counts


def _erase_row_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, _color in bars:
            out[row, cursor - length : cursor] = 0
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, _color in bars:
            out[row, cursor : cursor + length] = 0
            cursor += length


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    combo_out = main(inp)
    trail_only_out = _solve_variant(inp, _paint_periodic_trails, erase_source=False)
    erase_only_out = _solve_variant(inp, _paint_periodic_points, erase_source=True)
    base_out = _solve_variant(inp, _paint_periodic_points, erase_source=False)

    sep_col, _ = _find_sep_col(inp, 0)
    if sep_col is None:
        return {
            "diff_vs_base": 0,
            "diff_vs_trail_only": 0,
            "diff_vs_erase_only": 0,
            "long_bars": 0,
            "max_bar_length": 0,
            "ext_nonzero": 0,
        }

    left_mass = int(np.sum(inp[:, :sep_col] != 0)) if sep_col > 0 else 0
    right_mass = int(np.sum(inp[:, sep_col + 1 :] != 0)) if sep_col + 1 < inp.shape[1] else 0
    bars_on_left = left_mass >= right_mass
    bars = _collect_bars(inp, sep_col, bars_on_left, 0)
    lengths = [length for _dist, _r, _c0, _c1, length, _col in bars]
    ext_slice = combo_out[:, sep_col + 1 :] if bars_on_left else combo_out[:, :sep_col]

    return {
        "diff_vs_base": int(np.sum(combo_out != base_out)),
        "diff_vs_trail_only": int(np.sum(combo_out != trail_only_out)),
        "diff_vs_erase_only": int(np.sum(combo_out != erase_only_out)),
        "long_bars": int(sum(length >= 4 for length in lengths)),
        "max_bar_length": int(max(lengths) if lengths else 0),
        "ext_nonzero": int(np.sum(ext_slice != 0)),
    }


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
    active_row_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_row_set]

    latent = np.zeros((h, w), dtype=int)
    latent[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        latent[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    long_row = int(rng.choice(active_rows))
    row_bars_map: dict[int, tuple[list[tuple[int, int]], int]] = {}

    for idx, row in enumerate(active_rows):
        sample = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            require_long=(row == long_row),
        )
        if sample is None:
            return None
        bars, sep_gap = sample
        if any(added <= 0 for added in _bar_visibility_counts(w, sep_col, bars_on_left, bars)):
            return None
        row_bars_map[row] = (bars, sep_gap)
        _place_bars(latent, row, sep_col, bars_on_left, bars, sep_gap)

    for row, (bars, _sep_gap) in row_bars_map.items():
        _paint_extension(latent, row, sep_col, bars_on_left, bars)

    inp = latent.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    out = latent.copy()
    for row, (bars, sep_gap) in row_bars_map.items():
        _erase_row_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    metrics = _variant_metrics(inp)
    if metrics["long_bars"] <= 0 or metrics["max_bar_length"] < 4:
        return None
    if metrics["diff_vs_erase_only"] <= 0:
        return None
    if metrics["diff_vs_trail_only"] <= 0:
        return None
    if metrics["diff_vs_base"] <= metrics["diff_vs_trail_only"]:
        return None
    if metrics["ext_nonzero"] < 4:
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
