from __future__ import annotations

# Variant of 1ae2feb7
# Dimension: D1 — 射线路径方向 (path)
# Change: straight_ray——从分隔条出发，沿水平方向向对侧网格边缘延伸（垂直于分隔条方向的直线射线）... → 双向射线
#
# Design analysis
# a) 具体化与行为追踪
# - 仍然有一根几乎贯穿全高的竖直分隔条，以及全部位于同一侧的水平单色条。
# - 原题里，每个条只会在“分隔条另一侧”按步长=条长周期打点；本变体里，同一个条还会在
#   “自己的那一侧”继续向远离分隔条的方向按同样周期打点。
# - 这里把“双向”具体化为：对某条长度 L 的条，仍按“离分隔条最近优先、不覆盖已有颜色”
#   处理；若条在左侧，则扫描列序列 `sep+1, sep+1+L, ...` 与 `sep-1, sep-1-L, ...`；
#   若条在右侧，则扫描 `sep-1, sep-1-L, ...` 与 `sep+1, sep+1+L, ...`。
# - 小例子：左侧有一条长度 3、颜色 7 的条，占据 `sep-3:sep-1`。原题只会把右侧
#   `sep+1, sep+4, sep+7, ...` 涂成 7；本变体还会检查左侧 `sep-1, sep-4, sep-7, ...`。
#   其中 `sep-1` 已被原条占据，不变；若 `sep-4` 是背景，就会新增一个 7。差异正体现在
#   bar_side 外缘背景区也被周期点阵侵入。
# b) 复杂度校准
# - 不引入新的对象类别，也不改变“同一行 1-2 条、按近远顺序处理”的主逻辑，避免规则爆炸。
# - 为了让变体肉眼可见，生成器强制每个活动行在 bar_side 最外侧至少保留 1 格背景；否则
#   双向射线可能完全落在原条上，视觉上退化成原题。
# - 仍保留左右翻转、1/2 条、近端可留 1 格 gap、不同条长与颜色等自由度，但都属于原题已
#   存在的结构性变化，样例仍可由 2-4 组归纳。
# c) 原 main() 中与变体相关的隐式假设审查
# - `_find_sep_col` 只依赖“存在几乎满列的单色竖条”，变体未改，仍成立。
# - `bars_on_left = left_mass >= right_mass` 依赖输入里只有 bars 一侧非空。若把 bar_side
#   新增涂点也留在输入中，这个判断就会失真；所以生成器必须继续让输入只包含原 bars。
# - `bars.sort(key=(dist,row,col))` 依赖“离分隔条近的条先处理，且不覆盖已有颜色”。
#   变体仍保留这一优先级，因为双向延伸也会在 bar_side 发生冲突。
# - 原 solver 最关键的旧假设是“只需要向 ext_side 扫描一侧”。这在变体中不再成立，所以
#   唯一必要改动就是把每条 bar 的扫描范围扩成两个方向；其它 bar 抽取与排序逻辑可复用。
# d) 放置合法性
# - 生成器继续保证所有 bars 都在分隔条同侧、同一活动行的两条 bar 相邻、总长度不越界。
# - 额外加入 `OUTER_MARGIN_MIN = 1`，即 `sum(lengths) + sep_gap <= bar_side_len - 1`，
#   以确保 bar_side 上至少有 1 格可被新射线涂到的背景。
# - 输入单独构造为“背景 + separator + bars”；输出则在其副本上模拟双向射线，因此不会把
#   输出中的新增点误带回输入。
# e) 视觉可观察性
# - 排序完全基于空间距离与行内位置，不使用颜色 id 做任何不可见决策。
# - 从样例上，人会看到原本只在 ext_side 出现的周期点阵，现在在 bars 那一侧的外缘背景区
#   也出现，同一行呈现从分隔条向左右两边发散的节奏。
# f) 可视化整洁度
# - 活动行之间至少隔 1 空行，避免双向延伸把整张图糊成一片。
# - 条数仍限 1-2，且 bar_side 强制保留少量外缘背景，能看清“原条主体”和“新增周期点”。
# g) 颜色变化可推断性
# - 不新增颜色映射；新增像素直接继承对应 bar 的颜色。解题者只需识别“哪条的长度控制哪组
#   周期点”，而不必再推断颜色替换。
# h) 参考任务
# - source_task 为 None，本变体不需要额外参考别的 solver。
# i) 修改后 main() 的新假设与生成器保证
# - 新 solver 假设输入 ext_side 仍为空，否则可能把已有非背景误当成“不可覆盖障碍”。
#   生成器通过只输出原始输入结构保证这一点。
# - 新 solver 还假设 bar_side 存在至少一部分背景，才能稳定观察到新行为。生成器用外缘保留
#   约束保证这点，但 solver 本身对“没有可涂位置”的输入也能安全退化处理。

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


def _paint_bidirectional_ray(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    length: int,
    color: int,
    bg: int,
) -> None:
    """Paint the periodic ray on both sides of the separator without overwriting."""
    w = out.shape[1]
    if bars_on_left:
        scans = (
            range(sep_col + 1, w, length),
            range(sep_col - 1, -1, -length),
        )
    else:
        scans = (
            range(sep_col - 1, -1, -length),
            range(sep_col + 1, w, length),
        )
    for positions in scans:
        for cc in positions:
            if int(out[row, cc]) == bg:
                out[row, cc] = color


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

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        _paint_bidirectional_ray(out, r, sep_col, bars_on_left, length, col, bg)

    return out


# --------------- generator ---------------

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

OUTER_MARGIN_MIN = 1
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
    """Return (bars, sep_gap) with bars ordered near->far."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap - OUTER_MARGIN_MIN
    if capacity < 1:
        raise ValueError("capacity must leave at least one outer background cell")

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


def _place_bars(
    inp: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    """Paint the visible input bars on the bar side, from near to far."""
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, color in bars:
            inp[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            inp[row, cursor : cursor + length] = color
            cursor += length


def _simulate_output(
    inp: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    row_bars_map: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    out = inp.copy()
    for row, bars in row_bars_map.items():
        for length, color in bars:
            _paint_bidirectional_ray(out, row, sep_col, bars_on_left, length, color, bg=0)
    return out


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

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        inp[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        row_bars_map[row] = bars
        _place_bars(inp, row, sep_col, bars_on_left, bars, sep_gap)

    out = _simulate_output(inp, sep_col, bars_on_left, row_bars_map)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), out):
        return None

    # Reject degenerate samples where the new D1 behavior is invisible on the bar side.
    if bars_on_left:
        bar_side_added = int(np.sum((inp[:, :sep_col] == 0) & (out[:, :sep_col] != 0)))
    else:
        bar_side_added = int(np.sum((inp[:, sep_col + 1 :] == 0) & (out[:, sep_col + 1 :] != 0)))
    if bar_side_added <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
