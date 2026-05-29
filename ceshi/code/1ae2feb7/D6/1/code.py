# Variant of 1ae2feb7
# Dimension: D6 — 输出网格尺寸 (grid_shape)
# Change: same——输出与输入同尺寸，仅在ext_side区域新增涂色内容... → 裁剪为仅活跃行
#
# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 输入仍是原题版式: 一整列竖直分隔条, 所有 bar 都在同一侧, ext_side 在输入中全为背景。
#    - 输出先按原题规则在 ext_side 做周期涂点, 但最终不保留整张高 H 的网格, 只保留“含有至少一个 bar 的行”。
#    - 这些活跃行的左右宽度完全不变: bar_side、separator、ext_side 都保留, 只是把上下空白行删掉。
#    - 小例子: 若输入高度 9, 活跃行为 [1, 4, 7], 原题输出仍高 9; 本变体先得到同样的高 9 着色结果,
#      再返回第 1/4/7 行组成的 3xW 网格。差异只发生在“最终输出装配”阶段, 不是 bar 传播规则本身。
# b) 复杂度校准
#    - 不改 bar 的数量、相对顺序、颜色或周期逻辑, 只改输出高度, 保持和原题同级复杂度。
#    - 原生成器已保证活跃行之间至少隔 1 个空白行, 且高度 8-13、活跃行仅 2-5, 因而裁剪几乎总是明显可见。
# c) 隐式假设审查（对应 main 的关键依赖）
#    - `_find_sep_col`: 依赖“分隔条是一列非背景同色、覆盖至少 H-1 行”。D6 不改变输入, 该前提仍成立。
#    - `bars_on_left` 通过比较分隔条左右非背景像素数判断 bar 所在侧。裁剪只影响输出, 输入统计不变。
#    - `_iter_runs_1d` 与逐行收集 bars 依赖“bar 是单行水平纯色 run”。D6 不改变 bar 几何, 仍成立。
#    - 周期涂点循环依赖“按离分隔条最近优先、且不覆盖已有颜色”。D6 不改变此顺序, 仍先生成完整输出。
#    - 新增假设: 裁剪依据是“含 bar 的行”, 不是“separator 非空的行”。因此必须从收集到的 bars 推出 active_rows,
#      不能用分隔条列非零行替代, 否则会错误保留只含 separator 的空行。
# d) 放置合法性
#    - generator 仍采用 output-first: 先构造完整高 H 的期望输出, 再清空 ext_side 得输入。
#    - bars 的总长度、与 separator 的 0/1 间隔、左右侧宽度约束都沿用原题, 因而无越界/重叠新增风险。
#    - 最终裁剪只抽取已存在的活跃行子序列, 不会引入覆盖或位置冲突。
# e) 视觉可观察性
#    - 解题者会看到: 输入里上下有很多空白行和完整分隔条, 输出却只剩下有 bar 的几行, 宽度不变。
#    - 因为活跃行之间本来有空白间隔, 裁剪后的高度压缩在 2-3 个样例中就很明显, 不依赖颜色 id 等隐式因素。
# f) 可视化整洁度
#    - 裁剪后每一行仍只包含 bar、separator、周期点三层结构, 没有新增叠加。
#    - 删除空白行反而让输出更紧凑, 不会增加视觉混乱。
# g) 颜色变化可推断性
#    - 本变体不引入新的颜色映射。颜色仍只承担原题中的 bar/周期传播角色, 推断负担不增加。

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


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
    g: np.ndarray, sep_col: int, bg: int, bars_on_left: bool
) -> list[tuple[int, int, int, int, int, int]]:
    h, w = g.shape
    bars: list[tuple[int, int, int, int, int, int]] = []
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
    return bars


def _crop_to_active_rows(g: np.ndarray, active_rows: list[int]) -> np.ndarray:
    if not active_rows:
        return g.copy()
    return g[np.array(active_rows, dtype=int), :].copy()


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

    bars = _collect_bars(g, sep_col, bg, bars_on_left)

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        if bars_on_left:
            for cc in range(sep_col + 1, w, length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col

    active_rows = sorted({r for _, r, _, _, _, _ in bars})
    return _crop_to_active_rows(out, active_rows)


# --------------- generator (D6 crop-active-rows variant) ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5

P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 300


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
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = color


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

    full_out = np.zeros((h, w), dtype=int)
    full_out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        full_out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        row_bars_map[row] = bars
        _place_bars(full_out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension(full_out, row, sep_col, bars_on_left, bars)

    inp = full_out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    expected = _crop_to_active_rows(full_out, active_rows)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if expected.shape[0] >= h:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
