# Variant of 1ae2feb7
# Dimension: D5 — 变换/输出操作 (transform)
# Change: 无显式变换——仅在ext_side的背景格子上涂色，不修改已有的分隔条或色条
#         → 涂色后擦除源色条

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the D5 variant
# ==================================
# a) 具体化与行为追踪
# - 输入仍与原题一致：一根几乎贯穿全高的竖直分隔条，把网格分成 bar_side 和 ext_side；
#   若干活动行在 bar_side 上有 1-2 条水平纯色色条，ext_side 在输入中保持背景色。
# - 输出先执行原题的周期性涂点：每条色条以“长度 = 步长”的方式，从紧邻分隔条的 ext_side
#   起点开始向外每隔 length 涂一个点，按“离分隔条近的色条先处理，且不能覆盖已有颜色”。
# - 新变化：ext_side 涂完以后，把输入中 bar_side 上的原始色条整段擦成背景色。输出最终只保留：
#   分隔条 + ext_side 上的周期性涂点；bar_side 恢复为空白。
# - 小例子（bars_on_left=True, sep_col=4）：
#   输入某行为 [0,2,2,3,9,0,0,0,0,0]
#   其中 9 是分隔条，近条为颜色 3、长度 1，远条为颜色 2、长度 2。
#   原题输出：近条先在列 5,6,7,8,9 涂 3，远条尝试列 5,7,9 但这些位置已非背景，故不覆盖。
#   变体输出：ext_side 仍得到 [3,3,3,3,3]，但 bar_side 上原来的 [2,2,3] 被擦成 0，
#   整行变为 [0,0,0,0,9,3,3,3,3,3]。
# - 差异只发生在输出末尾；输入构造、分隔条识别、近到远顺序、no-overwrite 规则都不变。
#
# b) 复杂度校准
# - 没有引入新的对象类型，也没有引入按颜色 id 排序等不可见机制。
# - 视觉差异是稳定且显著的：每个活动行在输出中都会“失去左/右侧源条，只留下另一侧的点列”。
# - 原题已有 1-2 条色条、不同长度、可选 1 格分隔 gap，这些自由度足够支撑人类从少量样例
#   看出“这些点来自已被擦除的源条”，无需额外复杂化。
#
# c) 隐式假设审查（针对 main 的相关代码）
# - _find_sep_col 依赖“存在一列非背景像素几乎全同色”，D5 不改变分隔条，因此仍成立。
# - left_mass/right_mass 用输入判断哪侧是 bar_side；因为输入仍保留源条、ext_side 仍为空，所以成立。
# - _iter_runs_1d 依赖 bar_side 上对象是水平同色 run；D5 不改变输入结构，因此成立。
# - bars.sort(key=(dist,row,col)) 依赖“近条优先”决定 ext_side 遮挡关系；D5 不改变这一点。
# - 原版 `out = g.copy()` 默认保留源条；在变体中这会错误地把 bar_side 留在输出中。
#   修正方式是：继续基于 `g.copy()` 做涂点，保证分隔条和其它背景保持原状；但在完成全部涂点后，
#   按原始 bars 记录的坐标统一擦除 bar pixels。若边涂边擦，会破坏后续仍需读取的长度/颜色来源。
#
# d) 放置合法性
# - generator 仍遵循原 spec：活动行之间至少隔 1 空行；每行 1-2 个相邻色条，总长度不超过 bar_side。
# - ext_side 涂点只发生在 separator 另一侧，天然不会与源条重叠；擦除步骤只作用于 bar_side 上
#   已记录的 run 坐标，也不会越界或误擦 separator。
# - 生成仍采用 output-first：先构造“带源条 + ext 点列”的潜在完整态，再派生
#   input=清空 ext_side，output=清空 bar_side，并用 main(input) 校验。
#
# e) 视觉可观察性
# - 解题者在 2-3 组样例中能直接看到：输入里有靠近分隔条的横条，输出里横条消失，而分隔条另一侧
#   出现周期性点列；这是肉眼可见的结构变化，不依赖隐藏属性。
# - 关键分支（左侧条 / 右侧条，单条 / 双条，有无 1 格 gap）继续随机出现，避免样例长期固定成单一版式。
#
# f) 可视化整洁度
# - 输出比原题更干净：bar_side 被清空后，同一区域不会同时堆叠“源条 + 派生标记”。
# - ext_side 仍只承载点列，no-overwrite 保持原题整洁的遮挡规则，不会制造多层混色。
#
# g) 颜色变化可推断性
# - 没有颜色映射。颜色仅从源条复制到 ext_side，对应关系由同一行的长度/位置结构提供线索。
# - 颜色种类延续原题的 5-8 色调色板，但每行只实际出现 1-2 种条色，推断负担可控。


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

    # New assumption audit:
    # - This side inference still depends on INPUT mass, not output mass.
    # - The generator guarantees ext_side is blank in the input, so whichever side has more
    #   non-bg pixels must be the source-bar side.
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

    # Preserve the original nearest-first extension logic.
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, r, _c0, _c1, length, col in bars:
        if bars_on_left:
            for cc in range(sep_col + 1, w, length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col

    # D5 change: erase the original source bars after their contribution has been projected.
    for _, r, c0, c1, _length, _col in bars:
        out[r, c0 : c1 + 1] = bg

    return out


# --------------- generator (D5 erase-source-bar variant) ---------------

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
    out: np.ndarray,
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
    """Simulate the periodic extension onto the ext side (nearest-first, no overwrite)."""
    w = out.shape[1]
    for length, color in bars:
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = color


def _erase_source_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    """Erase the original source bars while leaving the separator and ext-side marks intact."""
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

    latent = np.zeros((h, w), dtype=int)
    latent[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        latent[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, tuple[list[tuple[int, int]], int]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
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
        _erase_source_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    erased_pixels = int(np.sum((inp != 0) & (out == 0)))
    if erased_pixels <= 0:
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
