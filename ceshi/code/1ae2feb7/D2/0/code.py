# Variant of 1ae2feb7
# Dimension: D2 — 轨迹涂色模式 (trajectory)
# Change: periodic_pattern——以间隔=色条像素数在ext_side上涂单色点（位置0, length, 2*len...）→ 连续轨迹填充

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D2 variant
# ----------------------------------
# variant_spec:
#   unchanged_structure:
#     - 输入仍然是单个竖直分隔列，把网格分成 bar_side 和 ext_side。
#     - bar_side 上仍然只有 2-5 个活动行；每个活动行有 1-2 个贴近分隔列的水平纯色色条。
#     - 若一行有两个色条，仍按“离分隔列最近的色条先处理，且不覆盖已有颜色”。
#   changed_behavior:
#     - 原题：每个色条长度 = interval，只在 ext_side 的相对位置 0, interval, 2*interval, ... 上涂 1 个像素。
#     - 变体：同样以色条长度作为 interval，但在每个起点位置涂一段长度 floor(interval/2) 的连续色段。
#     - 右扩展时段方向向右，左扩展时段方向向左；仍然逐像素遵守“不覆盖已有颜色”。
#   concrete_example:
#     - 若分隔列在中间、某行最近色条长度为 4、颜色为红：
#       原题输出的 ext_side 图案是 `R . . . R . . . R ...`
#       变体输出改成 `R R . . R R . . R R ...`
#     - 若同一行更远色条长度为 3、颜色为蓝：
#       它尝试在 `0, 3, 6, ...` 起段，但每段长度 floor(3/2)=1，只能在最近色条未占用的位置补点。
#       因而“近条先铺宽段，远条补空位”的遮挡关系仍保留。
#   flow_impact:
#     - 输入构造不变：仍然先生成完整输出，再把 ext_side 清空得到输入。
#     - solver 里只有“ext_side 绘制器”变化；色条检测、分隔列检测、最近优先顺序都不需要改。
#   complexity_calibration:
#     - 如果允许大量长度 1/2/3 的色条，则 floor(length/2) 仍为 0/1，输出会退化为原始稀疏点阵或几乎无变化。
#     - 因此生成器需要稳定地产生至少一个长度 >= 4 的色条，并过滤掉新旧规则输出完全相同的样本。
#     - 其余自由度仍保持原题规模：左右分布、活动行数、1-2 条色条、可选 1 格分隔空隙都保留。
#   implicit_assumption_audit:
#     - `_find_sep_col` 假设只有分隔列会在 >=H-1 个格子里呈现同色非背景。变体不改变这一点。
#     - `_iter_runs_1d` 假设同色相邻像素属于同一条 bar。生成器继续保证同一行双 bar 颜色不同，避免被合并。
#     - `bars.sort(key=(dist,row,col))` 依赖“近条优先”这一旧规则；变体仍保持，所以排序可复用。
#     - 原题单点绘制循环只写一个坐标；变体若沿用该写法会漏掉连续段，因此必须显式遍历/切片整段。
#     - 新规则下 `trail_len = length // 2` 可能为 0；solver 对此安全地跳过。生成器则负责提供足够多的 `length>=4`
#       样本，避免视觉退化。
#   placement_legality:
#     - ext_side 仍然只按单行水平轨迹着色，不会跨行，因此与其他活动行不相撞。
#     - 同一行的重叠只来自多条 bar 对同一 ext_side 的竞争；solver 和 generator 都按“近条先、不得覆盖”逐像素处理。
#     - 色条仍完全放在 bar_side 内，双 bar 总长度不超过 bar_side 宽度，且最多只允许与分隔列之间留 1 格空隙。
#   visual_observability:
#     - 这个变体的可见差异不是颜色 id，而是条纹密度：长色条会在 ext_side 产生成段的短横条，而不是离散点。
#     - 生成器强制保留至少一个 `length>=4` 的 bar，并要求新旧规则输出不同，从而确保 2-3 个样例就能看出差异。
#   neatness:
#     - 轨迹仍然局限在各自的活动行，行间隔至少 1 个空行，避免 dense stripes 彼此粘连成大块。
#   new_solver_assumptions_checked_against_generator:
#     - solver 假设输入里至少存在一个可识别的竖直分隔列；generator 继续整列填充，并只允许最多一个非活动行缺口。
#     - solver 假设 bar_side 全在分隔列同一侧；generator 不变。
#     - solver 假设 ext_side 输入全空；generator 在最后统一清空 ext_side，满足这一点。


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
            row = g[r, :]
            for c0, c1, col in _iter_runs_1d(row, 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, c1, length, col))
        else:
            row = g[r, :]
            for c0, c1, col in _iter_runs_1d(row, sep_col + 1, w, bg=bg):
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
    h, w = out.shape
    del h
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


def _solve_with_painter(grid: np.ndarray, painter) -> np.ndarray:
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
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_painter(grid, _paint_periodic_trails)


# --------------- generator (D2 continuous-trail variant) ---------------

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

    if require_long:
        if capacity == 4 or rng.random() < 0.65:
            length = int(rng.integers(4, capacity + 1))
            return [(length, primary)], sep_gap
        others = [c for c in palette if c != primary]
        long_near = bool(rng.integers(0, 2) == 0)
        long_len = int(rng.integers(4, capacity))
        other_len = int(rng.integers(1, capacity - long_len + 1))
        other_color = int(rng.choice(others))
        if long_near:
            return [(long_len, primary), (other_len, other_color)], sep_gap
        return [(other_len, other_color), (long_len, primary)], sep_gap

    two_bars = (capacity >= 2) and (rng.random() < P_TWO_BARS)
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


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    new_out = main(inp)
    old_out = _solve_with_painter(inp, _paint_periodic_points)
    sep_col, _ = _find_sep_col(inp, 0)
    if sep_col is None:
        return {"diff_cells": 0, "long_bars": 0, "max_bar_length": 0}
    left_mass = int(np.sum(inp[:, :sep_col] != 0)) if sep_col > 0 else 0
    right_mass = int(np.sum(inp[:, sep_col + 1 :] != 0)) if sep_col + 1 < inp.shape[1] else 0
    bars_on_left = left_mass >= right_mass
    bars = _collect_bars(inp, sep_col, bars_on_left, 0)
    lengths = [length for _dist, _r, _c0, _c1, length, _col in bars]
    return {
        "diff_cells": int(np.sum(new_out != old_out)),
        "long_bars": int(sum(length >= 4 for length in lengths)),
        "max_bar_length": int(max(lengths) if lengths else 0),
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
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    long_row = int(rng.choice(active_rows))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}

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
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension(out, row, sep_col, bars_on_left, bars)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    metrics = _variant_metrics(inp)
    if metrics["long_bars"] <= 0 or metrics["max_bar_length"] < 4:
        return None
    if metrics["diff_cells"] <= 0:
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
