# Combo variant of 1ae2feb7 (index 14)
# combo_spec: D5/3+D6/1
# Name: 分隔条消失且裁剪活跃行

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


"""
design_analysis:
  a_specific_behavior:
    input_scene:
      - 输入仍保留原题骨架: 一根近乎贯穿全高的单色竖直分隔条, 所有 bars 都在同一侧, 另一侧 ext_side 全为空白.
      - solver 必须先从输入里定位分隔条, 再判断哪一侧是 bars, 再按“离分隔条最近优先且不覆盖”的原题顺序扩展周期点.
    output_scene:
      - 与原题不同, 最终输出不再保留整张 HxW 网格, 而是只保留含有 bar 的 active rows.
      - 同时, 这些保留下来的行里, separator column 会被统一擦成背景色.
      - 因而解题者看到的是一个压缩后的稠密网格: 左右宽度不变, 但没有分隔条, 也没有空白行.
    trace_example:
      - 设输入某行在 separator 左侧有一条长度 3 的 bar, 另一条活跃行有两段 bars 长度 2/1; 中间与上下夹着空白行.
      - 原题会先在完整高度网格上做右侧周期落点, 并保留 separator.
      - 融合版也先得到同样的逐行落点结果, 但随后只抽取那两条活跃行, 并把抽出的每行的 separator 位置清零.
      - 差异不只是“擦掉一列”或“删掉空白行”之一, 而是输出装配成一个没有显式分隔提示的紧凑结果.
  b_complexity_calibration:
    - 保留原题核心自由度: bars 可在左/右侧, 每行 1-2 段, 长度不同, 最近 bar 优先, 且可有 1 格 sep_gap.
    - 裁剪后高度降到 2-5 行, 视觉变化非常明显; 擦除分隔条又移除了最关键的支架, 难度明显高于任一单变体.
    - 规则仍只依赖可见结构关系, 不引入基于颜色 id 或隐藏状态的分支, 因此 2-4 个样例足以归纳.
  c_implicit_assumption_review_main:
    - `_find_sep_col` 依赖输入里仍存在近乎满列的单色 separator; 融合版不改输入, generator 继续保证这一点.
    - `bars_on_left` 依赖 ext_side 在输入中为空, 所以左右非背景像素质量能指示哪边是 bars; generator 继续清空 ext_side.
    - `_collect_bars` 依赖 bars 是单行水平纯色 run, 且都位于 separator 同一侧; 生成器保持该几何约束.
    - 周期扩展循环依赖“先按离分隔条距离排序, 再在背景格上落点”; 这一原题机制不变.
    - D5/3 破坏了“输出保留 separator”的旧假设, 所以需要显式清除 separator column.
    - D6/1 破坏了“输出高度等于输入高度”的旧假设, 所以不能直接返回 `out`; 必须从收集到的 bars 导出 active rows.
    - 组合后新增假设是: active rows 的判定标准必须是“该行含 bar”, 而不是“该行 separator 非空”, 否则裁剪会错误保留空行.
  d_placement_legality:
    - generator 采用 output-first: 先构造含 separator、bars、ext 落点的 latent full scene, 再派生输入与期望输出.
    - active rows 之间至少隔 1 空行, bars 总长度不超过 bar_side_len-sep_gap, 因而 bars 不重叠且不越界.
    - ext_side 只在背景格落点, 所以不会覆盖已有颜色; 擦除 separator 发生在最终装配阶段, 也不会影响 bar 读取.
  e_visual_observability:
    - 人类会稳定看到: 输入里有整列 separator 和很多空白行, 输出却只剩几条有 bar 的行, 且原 separator 那一列变成背景.
    - 因为裁剪和擦除同时发生, 输出既失去竖向定位支架又失去行间空白, 融合效果在 2-3 个样例上就足够显眼.
  f_visual_cleanliness:
    - 虽然输出更紧凑, 但每一行仍只含三类信息: bar_side 的 bars、separator 位置留下的背景槽、ext_side 的周期点.
    - 行与行之间没有叠加; 删除空白行只压缩高度, 不会把不同对象混在同一行里.
  g_color_inference:
    - 不存在颜色映射, 所有 ext 点都沿用对应 bar 的原色.
    - 颜色数量仍限制在原题范围内, 且每行模式由 bar 长度和相对位置支撑, 人类可从结构中推断颜色传播关系.
  h_variant_metric:
    - 可量化指标 1: cropped_rows = 输入高度 - 输出高度, 本组合要求 cropped_rows >= 1.
    - 可量化指标 2: erased_separator_cells = 输出中 separator 列被清零的 active-row 单元数, 本组合要求该值等于输出高度且 >= 2.
    - 若任一指标失效, 就说明样本退化为未裁剪或未擦除分隔条的旧行为.
  i_new_main_assumptions:
    - `main` 新增“输出仅由含 bar 的行组成”的假设; generator 保证 active row 数至少 2, 且非 active rows 不含 bar.
    - `main` 还假设 separator 可先被识别、后被擦除; generator 保证 separator 颜色独立且该列至少有 h-1 个同色非背景像素.
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


"""
Overview:
Detect the separator and bars exactly as in the original task, paint the periodic
extension on the opposite side, then erase the separator and compress the output
to only the rows that actually contain bars.

Input structure:
- A mostly full-height monochrome vertical separator column divides the grid.
- All horizontal monochrome bars lie on one side of that separator.
- The other side is blank in the input and serves as the periodic painting area.

Rule:
- For each bar, process rows by bar distance to the separator, nearest first.
- Paint the opposite side at period = bar length, starting from the cell adjacent
  to the separator, without overwriting non-background cells.
- After all periodic painting, erase the separator column to background.
- Finally keep only the active rows that contained at least one input bar.

Key insight:
The output no longer shows either of the original structural scaffolds: the
separator itself is gone, and the blank spacer rows are removed, so the solver
must recover both from the input before assembling the compact final view.
"""
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)

    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    bars = _collect_bars(g, sep_col, bg, bars_on_left)
    if not bars:
        return g.copy()

    out = g.copy()
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

    out[:, sep_col] = bg
    active_rows = sorted({r for _, r, _, _, _, _ in bars})
    return _crop_to_active_rows(out, active_rows)


# --------------- generator (combo D5/3 + D6/1) ---------------

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
    """Return (bars, sep_gap) where bars is ordered near->far."""
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


def _paint_extension(
    scene: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    w = scene.shape[1]
    for length, color in bars:
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if scene[row, cc] == 0:
                scene[row, cc] = color


def _clear_ext_side(scene: np.ndarray, sep_col: int, bars_on_left: bool) -> np.ndarray:
    inp = scene.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0
    return inp


def _build_expected(scene: np.ndarray, sep_col: int, active_rows: list[int]) -> np.ndarray:
    out = scene.copy()
    out[:, sep_col] = 0
    return _crop_to_active_rows(out, active_rows)


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

    scene = np.zeros((h, w), dtype=int)
    scene[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        scene[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        row_bars_map[row] = bars
        _place_bars(scene, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension(scene, row, sep_col, bars_on_left, bars)

    inp = _clear_ext_side(scene, sep_col, bars_on_left)
    expected = _build_expected(scene, sep_col, active_rows)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if np.count_nonzero(inp[:, sep_col] == sep_color) < h - 1:
        return None
    if expected.shape[0] >= h:
        return None
    if expected.shape != (active_count, w):
        return None
    if np.any(expected[:, sep_col] != 0):
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
