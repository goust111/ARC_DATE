# Variant of 1ae2feb7
# Dimension: D5 — 变换/输出操作 (transform)
# Change: 无显式变换——仅在ext_side的背景格子上涂色，不修改已有的分隔条或色条... → 分隔条消失

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


"""
design_analysis:
  a_specific_behavior:
    input_scene:
      - 输入仍包含一个几乎贯穿全高的单色竖直分隔条, 一侧是1-2段水平纯色色条, 另一侧全为背景.
      - solver 必须先靠这根竖条定位 sep_col, 再读取每个 active row 上靠近分隔条的 bars.
    output_scene:
      - ext_side 的周期性涂点规则完全继承原题: 每个 bar 以自身长度为步长, 从紧邻分隔条的第一格开始向外落点.
      - 变化只发生在最后一步: 输出中整个 separator column 被擦成背景色, bars 和 ext_side 点保留.
    trace_example:
      - 若某行左侧 bars 为 [length=2,color=1] 紧邻分隔条, 原题输出会在右侧第1/3/5...格涂1, 且分隔条保留.
      - 本变体先得到相同的右侧涂点, 然后把分隔条整列改成背景, 因此最终只剩左侧 bar 和右侧点阵.
      - 差异集中在最终 render 阶段, 不改变 bar 提取和涂点顺序.
  b_complexity_calibration:
    - 不增加新对象, 只移除输出中的视觉支架, 难点来自“输入可见、输出消失”的对照.
    - 保留原题的左右朝向、1或2条 bars、长度差异、可选1格 sep_gap, 让少量样例中仍有足够重复支撑.
    - 不引入基于颜色 id 的排序或额外分支; 最近 bar 优先这一可见结构保持不变.
  c_implicit_assumption_review_main:
    - _find_sep_col 依赖“输入里存在近乎整列的单色竖条”; 变体没有修改输入, 因此前提仍成立.
    - left_mass/right_mass 依赖“ext_side 在输入中为空”; generator 继续保证这一点.
    - bars.sort(key=(dist,row,c0)) 依赖“处理优先级由与分隔条距离决定”; 该机制仍是规则核心.
    - 原代码 out = g.copy() 默认保留 separator; 这正是变体需要修正的旧假设.
    - 若一开始就擦掉 separator, side detection 已完成但最终与原题等价; 更清晰安全的做法是涂点后统一清除整列.
  d_placement_legality:
    - generator 仍沿用 output-first 思路, 但内部先构造 latent scene = separator + bars + ext dots.
    - input 从 latent scene 派生: 清空 ext_side, 保留 separator 与 bar_side.
    - expected output 从 latent scene 派生: 清空 separator column.
    - bars 的总长度仍不超过 bar_side_len-sep_gap, 因此不会越界或互相重叠.
    - ext_side 只在背景格落点, 因此不会覆盖已有颜色.
  e_visual_observability:
    - 人类从样例可直接看到: 输入中竖条存在, 输出中该列变空, 同时另一侧出现原题式周期点.
    - 该变化在2-3个样例中即可稳定观察, 不依赖隐藏变量.
  f_visual_cleanliness:
    - 分隔条被擦除后, 输出更简洁, 只剩 bar_side 结构与 ext_side 点阵, 不会引入额外叠层.
    - active rows 保持至少隔1空行, 避免不同 row 的点阵视觉混杂.
  g_color_inference:
    - 颜色不发生映射, 只继承 bar 的原色; 颜色种类上限保持原题范围.
  h_variant_metric:
    - 可量化指标是 erased_separator_cells = 输入中 separator 列的非背景像素数.
    - 对非退化样本应满足 erased_separator_cells >= h-1, 且新 solver 输出与原 solver 输出至少在 separator 列不同.
  i_new_main_assumptions:
    - main 新增“整列擦除”步骤, 假设 separator 不是 bar 的一部分; generator 保证 bars 永不进入 separator column.
    - main 仍假设唯一可辨识分隔条足够显著; generator 通过专用 sep_color 和 >=h-1 的竖列保障这一点.
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

    bars = []
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
        if bars_on_left:
            for cc in range(sep_col + 1, w, length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col

    out[:, sep_col] = bg
    return out


# --------------- generator (variant D5) ---------------

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
    """Pick separator color and a disjoint bar-color palette."""
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
    scene: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    """Paint bars on the bar side, stacking from nearest-to-separator outward."""
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
    """Simulate the periodic extension onto the ext side (nearest-first, no overwrite)."""
    w = scene.shape[1]
    for length, color in bars:
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if scene[row, cc] == 0:
                scene[row, cc] = color


def _build_output(scene: np.ndarray, sep_col: int) -> np.ndarray:
    out = scene.copy()
    out[:, sep_col] = 0
    return out


def _clear_ext_side(scene: np.ndarray, sep_col: int, bars_on_left: bool) -> np.ndarray:
    inp = scene.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0
    return inp


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
    expected = _build_output(scene, sep_col)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if np.count_nonzero(inp[:, sep_col] == sep_color) < h - 1:
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
