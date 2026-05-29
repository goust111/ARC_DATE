# Combo variant of 1ae2feb7 (index 27)
# combo_spec: D1/1+D5/3+D6/0
# Name: 双向射线无分隔裁剪

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


"""
Design analysis

a) 具体化与行为追踪
- 原题先读取 separator 一侧的水平 bar，然后只朝另一侧周期打点，最后保留整张图。
- 本融合版先沿用同样的“按离 separator 最近优先、步长=bar 长度、不覆盖已有颜色”
  的规则，但每条 bar 会向两个方向都打点：
  * ext_side：仍从 separator 另一侧的第一格开始。
  * bar_side：从 separator 同侧、紧邻 separator 的第一格开始，向远离 separator 的方向
    继续按同样步长落点。
- 完成双向打点后，输出阶段不再保留输入支架：
  * separator 整列擦成背景；
  * 原始 bar 本体也擦成背景；
  * 最后删除 separator 这一列，把左右两侧直接拼成宽度 `W-1` 的纯 pattern 网格。
- 小例子，bars 在左侧、separator 在第 5 列，某行有长度 3 的 bar 占据列 2..4：
  * 原题完整输出：右侧列 6,9,12... 出现该色，左侧 bar 保留不动。
  * 本题完整求解后：右侧仍在 6,9,12... 打点，左侧还会尝试 4,1,...；其中 4 被 bar 占用，
    1 若为空则涂色。
  * 最终输出：列 5 被删去，原 bar 的 2..4 也被清空，因此只剩左侧的周期点和右侧的周期点。

b) 复杂度校准
- 该设计同时保留原题的核心可见规律（长度决定周期、近端 bar 优先、不可覆盖），只把
  “作用区域”从单侧扩成双侧，并把最终可视对象压缩为纯点阵。
- 若 bar_side 没有任何新增点，则融合版会退化得接近“擦掉 bars 后只剩 ext_side 点阵”。
  因此生成器除了保留外缘背景，还会显式筛选样本，保证最终输出在原 bar_side 上也出现
  可见 pattern。
- 仍只保留左右朝向、1/2 条 bars、0/1 格 sep_gap 这些少量自由度，不再额外引入颜色映射
  或多阶段几何变换，避免 2-4 个例子无法归纳。

c) 隐式假设审查
- `_find_sep_col` 依赖“输入里有一根近乎满高的单色竖列”；融合版没有改输入结构，
  generator 继续保证 separator 至少有 `H-1` 个非背景像素。
- `bars_on_left = left_mass >= right_mass` 依赖“输入里只有一侧有 bars、另一侧为空”；
  generator 继续让输入只含 bars + separator，不把任何 pattern 预写入输入。
- bar 抽取逻辑依赖“同一行 1-2 个 bar，且都在 separator 同侧”；生成器保持这一点。
- 原 solver 的关键旧假设是：输出保留 bars 和 separator，且只需朝 ext_side 扫描。
  这在融合版里都不成立，所以 main 被拆成三步：
  1. 在完整输入上解析 bars。
  2. 在完整画布上执行双向射线。
  3. 擦除 separator 与原 bars，再删除 separator 列。
- 新假设：清除 bar 本体时，bar 只来自输入解析出的那些连续段，而不是由后续射线新增。
  generator 通过“输入 ext_side 为空、bar_side 只放原 bars”来保证这点可区分。

d) 放置合法性
- 生成器仍采用“先构造合法输入骨架，再模拟目标输出”的流程。
- 每个 active row 的 bars 总长度满足 `sum(lengths) + sep_gap <= bar_side_len - OUTER_MARGIN_MIN`，
  因而不会越界，并给 bar_side 留出至少一格非 bar 区域。
- 由于双向落点始终只写背景格，不会覆盖 separator、bars 或其它已落下的点。
- 输出裁剪只是删除 separator 列，不会改变其它像素的相对行序，也不会产生重叠。

e) 视觉可观察性
- 人类从样例上能同时看到三件事：
  * 输出宽度比输入少 1 列，说明 separator 被删掉了；
  * 输出里不再有原始横向实心 bar；
  * separator 两侧都出现了由 bar 长度控制间距的周期点。
- 这些现象都直接体现在像素布局上，不依赖颜色编号大小、生成顺序编号等不可视信息。

f) 可视化整洁度
- active rows 仍至少隔一空行，避免双向点阵上下串扰。
- 每行 bars 仍限制为 1-2 段，颜色直接继承 bar，不做颜色重映射，输出虽然更“空”，但仍
  能清楚看出每一行的周期结构来源。
- 通过删除 separator 与 bars，最终输出只保留 pattern，层次反而比“完整画布输出”更干净。

g) 变化的可推断性
- 不引入新的颜色语义。输出中的每个点仍然来自某个输入 bar 的原色。
- 颜色数量沿用原题上限，且活跃行之间通常有重复的“长度控制间距”结构，解题者主要推断
  几何周期规则，而不是颜色对应表。

h) 融合行为量化指标
- 对单个样本定义 `bar_side_pattern_cells`：
  输出中，原始 bar_side 那一半区域里剩下的非零像素数。
- 若该值为 0，则在裁剪并清除 bars 后，双向射线在 bar_side 上没有留下任何可见结果，
  样本会退化得过于接近单向版本。生成器因此强制该指标 > 0。

i) 新 main() 假设与生成器保证
- main 现在假设：在输入中，separator 左右能区分出唯一的“bar 一侧”；generator 继续保证
  ext_side 全空、bars 仅在一侧，满足该假设。
- main 还假设：删除原 bars 后，输出仍应在 bar_side 留下至少部分 pattern，才能体现融合。
  generator 通过行级可见性筛选和整图指标校验来保证这一点。
"""


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
    for r in range(g.shape[0]):
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, c1, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, g.shape[1], bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, c1, length, col))
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    return bars


def _paint_bidirectional_ray(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    length: int,
    color: int,
    bg: int,
) -> None:
    if length <= 0:
        return
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


def _simulate_full(
    inp: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int, int, int, int, int]],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for _, r, _c0, _c1, length, col in bars:
        _paint_bidirectional_ray(out, r, sep_col, bars_on_left, length, col, bg)
    return out


def _finalize_output(
    full: np.ndarray,
    sep_col: int,
    bars: list[tuple[int, int, int, int, int, int]],
    bg: int,
) -> np.ndarray:
    out = full.copy()
    out[:, sep_col] = bg
    for _, r, c0, c1, _length, _col in bars:
        out[r, c0 : c1 + 1] = bg
    return np.concatenate((out[:, :sep_col], out[:, sep_col + 1 :]), axis=1)


def _row_bar_side_visibility(
    bar_side_len: int,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> int:
    """Return the visible pattern-cell count on the source side after bars are erased."""
    sep_col = bar_side_len
    row = np.zeros((1, bar_side_len + 1), dtype=int)
    cursor = sep_col - sep_gap
    records: list[tuple[int, int, int, int, int, int]] = []
    for length, color in bars:
        c0 = cursor - length
        c1 = cursor - 1
        row[0, c0 : c1 + 1] = color
        dist = (sep_col - 1) - c1
        records.append((dist, 0, c0, c1, length, color))
        cursor = c0
    records.sort(key=lambda t: (t[0], t[1], t[2]))

    full = row.copy()
    for _, _, _c0, _c1, length, color in records:
        _paint_bidirectional_ray(full, 0, sep_col, True, length, color, 0)
    final = _finalize_output(full, sep_col, records, 0)
    return int(np.count_nonzero(final))


# --------------- solver ---------------

# Overview:
# Build the original full solved scene first, but each bar now emits a periodic ray on
# both sides of the separator. The returned output is then converted into a pure-pattern
# crop by erasing the separator and original bars and deleting the separator column.
#
# Input structure:
# - A nearly full-height vertical separator column of one non-background color.
# - All horizontal monochrome bars lie on exactly one side of that separator.
# - The opposite side is background-only in the input.
#
# Rule:
# - Parse the bars in nearest-to-separator order.
# - For each bar, paint periodic dots with step = bar length on both sides of the
#   separator, never overwriting existing non-background cells.
# - After painting, erase the separator column and erase the original bar cells.
# - Finally remove the separator column from the grid, keeping both remaining sides.
#
# Key insight:
# The output is not the original scene with decorations; it is the latent bidirectional
# dot pattern with its generating scaffold removed.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    bars_on_left = left_mass >= right_mass

    bars = _collect_bars(g, sep_col, bars_on_left, bg)
    full = _simulate_full(g, sep_col, bars_on_left, bars, bg)
    return _finalize_output(full, sep_col, bars, bg)


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 4, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 4, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

OUTER_MARGIN_MIN = 1
MAX_ROW_BAR_TRIES = 80
MAX_ATTEMPTS = 500


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
) -> tuple[list[tuple[int, int]], int]:
    for _ in range(MAX_ROW_BAR_TRIES):
        sep_gap = 1 if rng.random() < P_SEP_GAP else 0
        capacity = bar_side_len - sep_gap - OUTER_MARGIN_MIN
        if capacity < 1:
            continue

        two_bars = capacity >= 2 and rng.random() < P_TWO_BARS
        if not two_bars:
            bars = [(int(rng.integers(1, capacity + 1)), primary)]
        else:
            near_len = int(rng.integers(1, capacity))
            far_len = int(rng.integers(1, capacity - near_len + 1))
            others = [c for c in palette if c != primary]
            colors = [primary, int(rng.choice(others))]
            if rng.random() < P_SWAP_COLORS:
                colors.reverse()
            bars = [(near_len, colors[0]), (far_len, colors[1])]

        if _row_bar_side_visibility(bar_side_len, bars, sep_gap) > 0:
            return bars, sep_gap
    raise RuntimeError("failed to sample a visible source-side row")


def _place_bars(
    scene: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, color in bars:
            scene[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            scene[row, cursor : cursor + length] = color
            cursor += length


def _segment_counts(expected: np.ndarray, sep_col: int, bars_on_left: bool) -> tuple[int, int]:
    if bars_on_left:
        bar_side = expected[:, :sep_col]
        ext_side = expected[:, sep_col:]
    else:
        ext_side = expected[:, :sep_col]
        bar_side = expected[:, sep_col:]
    return int(np.count_nonzero(bar_side)), int(np.count_nonzero(ext_side))


def _count_erased_source_cells(
    inp: np.ndarray,
    expected: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
) -> int:
    count = 0
    if bars_on_left:
        cols = range(0, sep_col)
    else:
        cols = range(sep_col + 1, inp.shape[1])
    for r in range(inp.shape[0]):
        for c in cols:
            if inp[r, c] == 0:
                continue
            out_c = c if c < sep_col else c - 1
            if expected[r, out_c] == 0:
                count += 1
    return count


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
    ext_side_len = (w - 1 - sep_col) if bars_on_left else sep_col
    if not (EXT_SIDE_MIN <= ext_side_len <= EXT_SIDE_MAX):
        return None

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

    bars = _collect_bars(inp, sep_col, bars_on_left, bg=0)
    full = _simulate_full(inp, sep_col, bars_on_left, bars, bg=0)
    expected = _finalize_output(full, sep_col, bars, bg=0)

    if expected.shape != (h, w - 1):
        return None
    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if np.count_nonzero(inp[:, sep_col] == sep_color) < h - 1:
        return None
    if int(np.count_nonzero(inp[:, :sep_col] if bars_on_left else inp[:, sep_col + 1 :])) <= 0:
        return None

    bar_side_cells, ext_side_cells = _segment_counts(expected, sep_col, bars_on_left)
    if bar_side_cells <= 0 or ext_side_cells <= 0:
        return None

    source_cells = int(np.count_nonzero(inp[:, :sep_col] if bars_on_left else inp[:, sep_col + 1 :]))
    if _count_erased_source_cells(inp, expected, sep_col, bars_on_left) != source_cells:
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
