# Variant of 1ae2feb7
# Dimension: D9 — 分隔条结构 (obstacle)
# Change: 单条垂直分隔列——一整列（允许缺1个格子）填充同色像素，将网格分为bar_side和ext_side
#   → 水平分隔行——一整行（允许缺1个格子）填充同色像素，将网格分为上下两半；
#   色条变为竖直单色色条，分布在分隔行的一侧，并向另一侧按“长度=步长”做垂直周期性涂点。
#
# Design analysis for the D9 variant
# ==================================
# 1. Concrete structure
#    - 原题的核心对象是“竖直分隔列 + 若干水平色条 + 向右/左周期性延伸”。
#    - 本变体将这套几何关系转置：输入里有一条水平分隔行；分隔行上方或下方是 bar_side，
#      另一侧是 ext_side。active columns 上放 1-2 段相邻的竖直单色色条，靠近分隔行的色条优先。
#    - 输出保持分隔行不变，只在 ext_side 中按竖直方向涂点：若某段色条长度为 L，则从紧邻分隔行
#      的第一个格开始，每隔 L 个格涂同色，且不覆盖已有颜色。
#
# 2. Behavior trace with a tiny example
#    - 原题某一 active row 里，长度 3 的近端条会在 separator 另一侧的同一行上涂到
#      col = sep+1, sep+4, sep+7 ...
#    - 本变体对应地在某一 active column 里，长度 3 的近端条会在 separator 另一侧的同一列上涂到
#      row = sep+1, sep+4, sep+7 ...
#    - 若同一列还有更远的一段长度 2 的条，则仍按“近端优先，不覆盖”处理，所以长度 2 只会填补
#      近端条没占到的周期位置。
#
# 3. Complexity calibration / observability
#    - 仍保留 1-2 段色条、2-5 个 active columns、近端可留 1 格缝隙，这样视觉上仍能看到：
#      分隔线、条段、周期点阵、近端优先覆盖。
#    - 为了让“转成上下结构”足够明显，生成器同步交换网格长宽范围：高度使用原题的 15-20，
#      宽度使用原题的 8-13。这样垂直延伸有足够长度，不会退化成只在分隔行旁边点一下。
#    - 生成器额外要求至少有 1 个 active column 在 ext_side 的“第二层及更远位置”出现涂点，
#      使人眼能直接看到垂直周期传播，而不是只有紧邻分隔行的一圈点。
#
# 4. Hidden assumptions audit for main()
#    - `_find_sep_col` 依赖“分隔条是列”；本变体改为 `_find_sep_row`。
#    - `left_mass/right_mass` 依赖条段只会出现在左右两侧；本变体改为 `top_mass/bottom_mass`。
#    - 逐行扫描 `_iter_runs_1d(g[r, :], ...)` 依赖条段是水平连续段；本变体改为逐列扫描
#      `_iter_runs_1d(g[:, c], ...)`，因为条段变成竖直连续段。
#    - 向 `cc` 方向扩展依赖“目标区域在左右方向”；本变体改为向 `rr` 方向扩展。
#    - 排序键 `(dist, r, c0)` 原本只要求“先按离 separator 的距离，再稳定遍历各行”。
#      本变体对应为 `(dist, c, r0)`，保证同样的近端优先与稳定性。
#
# 5. Placement legality guaranteed by generator
#    - separator row 的位置通过 `bar_side_len + 1 + ext_side_len = H` 约束，保证上下两侧都有空间。
#    - active columns 两两至少隔 1 个空列，避免不同列的条段在视觉上粘连。
#    - 同一列的两段条彼此相邻但不重叠，且总长度不超过 bar_side 高度容量。
#    - separator row 若出现 1 个缺口，只会放在 inactive column，避免破坏条段交点与识别。
#
# 6. New assumptions introduced by the rotated solver
#    - 输入中必须存在唯一可识别的水平分隔行：生成器保证只有 separator row 满足“非背景像素数>=W-1
#      且同色”。
#    - 所有竖直条段都在同一侧（上或下）：生成器统一采样 `bars_on_top`。
#    - ext_side 在输入里必须是空白：生成器仍采用 output-first，再清空 ext_side 反推输入。

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


def _find_sep_row(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = g.shape
    for r in range(h):
        row = g[r, :]
        vals = row[row != bg]
        if vals.size >= w - 1 and (vals == vals[0]).all():
            return r, int(vals[0])
    return None, None


def _iter_runs_1d(line: np.ndarray, i_lo: int, i_hi: int, bg: int):
    i = i_lo
    while i < i_hi:
        if int(line[i]) == bg:
            i += 1
            continue
        col = int(line[i])
        i0 = i
        while i < i_hi and int(line[i]) == col:
            i += 1
        yield i0, i - 1, col


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    sep_row, _sep_color = _find_sep_row(g, bg)
    if sep_row is None:
        return g.copy()

    out = g.copy()

    top_mass = int(np.sum(g[:sep_row, :] != bg)) if sep_row > 0 else 0
    bottom_mass = int(np.sum(g[sep_row + 1 :, :] != bg)) if sep_row + 1 < h else 0
    bars_on_top = top_mass >= bottom_mass

    bars = []
    for c in range(w):
        col = g[:, c]
        if bars_on_top:
            for r0, r1, color in _iter_runs_1d(col, 0, sep_row, bg=bg):
                length = r1 - r0 + 1
                dist = (sep_row - 1) - r1
                bars.append((dist, c, r0, r1, length, color))
        else:
            for r0, r1, color in _iter_runs_1d(col, sep_row + 1, h, bg=bg):
                length = r1 - r0 + 1
                dist = r0 - (sep_row + 1)
                bars.append((dist, c, r0, r1, length, color))

    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, c, _r0, _r1, length, color in bars:
        step = length
        if step <= 0:
            continue
        if bars_on_top:
            for rr in range(sep_row + 1, h, step):
                if int(out[rr, c]) == bg:
                    out[rr, c] = color
        else:
            for rr in range(sep_row - 1, -1, -step):
                if int(out[rr, c]) == bg:
                    out[rr, c] = color

    return out


# --------------- generator (D9 rotated variant) ---------------

# Swap the original aspect ranges so the 90-degree structural rotation remains visible.
H_MIN, H_MAX = 15, 20
W_MIN, W_MAX = 8, 13

# bar_side/ext_side now refer to heights above/below the separator row.
BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8
ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_ROW_GAP = 0.2

MAX_ATTEMPTS = 300


def _choose_active_cols(rng: np.random.Generator, w: int, count: int) -> list[int]:
    """Pick `count` column indices in [0, w) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(w - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_col_bars(
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
    col: int,
    sep_row: int,
    bars_on_top: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    """Paint bars on the bar side, stacking from nearest-to-separator outward."""
    if bars_on_top:
        cursor = sep_row - sep_gap
        for length, color in bars:
            out[cursor - length : cursor, col] = color
            cursor -= length
    else:
        cursor = sep_row + 1 + sep_gap
        for length, color in bars:
            out[cursor : cursor + length, col] = color
            cursor += length


def _paint_extension(
    out: np.ndarray,
    col: int,
    sep_row: int,
    bars_on_top: bool,
    bars: list[tuple[int, int]],
) -> None:
    """Simulate the periodic extension onto the ext side (nearest-first, no overwrite)."""
    h = out.shape[0]
    for length, color in bars:
        if bars_on_top:
            positions = range(sep_row + 1, h, length)
        else:
            positions = range(sep_row - 1, -1, -length)
        for rr in positions:
            if int(out[rr, col]) == 0:
                out[rr, col] = color


def _far_extension_stats(out: np.ndarray, sep_row: int, bars_on_top: bool) -> tuple[int, int]:
    """
    Quantify visible vertical propagation beyond the first row adjacent to the separator.
    Returns (far_nonzero_cells, columns_with_far_extension).
    """
    h, w = out.shape
    far_cells = 0
    far_cols = 0
    for c in range(w):
        if bars_on_top:
            if sep_row + 2 >= h:
                continue
            has_far = bool(np.any(out[sep_row + 2 :, c] != 0))
            if has_far:
                far_cells += int(np.sum(out[sep_row + 2 :, c] != 0))
                far_cols += 1
        else:
            if sep_row - 1 <= 0:
                continue
            has_far = bool(np.any(out[: sep_row - 1, c] != 0))
            if has_far:
                far_cells += int(np.sum(out[: sep_row - 1, c] != 0))
                far_cols += 1
    return far_cells, far_cols


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, h - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, h - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))

    bars_on_top = bool(rng.integers(0, 2) == 0)
    sep_row = bar_side_len if bars_on_top else (h - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (w + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_cols = _choose_active_cols(rng, w, active_count)
    inactive_cols = [c for c in range(w) if c not in set(active_cols)]

    out = np.zeros((h, w), dtype=int)
    out[sep_row, :] = sep_color
    if inactive_cols and rng.random() < P_SEP_ROW_GAP:
        gap_col = inactive_cols[int(rng.integers(0, len(inactive_cols)))]
        out[sep_row, gap_col] = 0

    col_primary = list(map(int, rng.permutation(palette)[:active_count]))
    col_bars_map: dict[int, list[tuple[int, int]]] = {}
    two_bar_cols = 0
    for idx, col in enumerate(active_cols):
        bars, sep_gap = _sample_col_bars(rng, palette, col_primary[idx], bar_side_len)
        if len(bars) == 2:
            two_bar_cols += 1
        col_bars_map[col] = bars
        _place_bars(out, col, sep_row, bars_on_top, bars, sep_gap)

    for col, bars in col_bars_map.items():
        _paint_extension(out, col, sep_row, bars_on_top, bars)

    far_cells, far_cols = _far_extension_stats(out, sep_row, bars_on_top)
    if far_cols < 1 or far_cells < 2:
        return None

    inp = out.copy()
    if bars_on_top:
        inp[sep_row + 1 :, :] = 0
    else:
        inp[:sep_row, :] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if two_bar_cols == 0 and active_count < 3:
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
