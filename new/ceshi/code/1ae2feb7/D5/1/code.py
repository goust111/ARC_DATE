# Variant of 1ae2feb7
# Dimension: D5 — 变换/输出操作 (transform)
# Change: 无显式变换——仅在ext_side的背景格子上涂色，不修改已有的分隔条或色条
#   → 完成 ext_side 周期性涂色后，将 bar_side 上所有原始色条统一改成分隔条颜色
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     - 保留原题唯一核心机制：按离分隔条的远近顺序处理 bar，周期 = bar 长度，只在 ext_side 的背景格上落点。
#     - 新增 D5 变体：ext_side 涂色完成后，再把 bar_side 的所有源 bar 像素重涂为 separator 的颜色。
#   behavior_trace:
#     - 原题单行示意：`[3 3 3 | 2 | 0 0 0 0 0] -> [3 3 3 | 2 | 3 0 0 3 0]`
#     - 变体单行示意：`[3 3 3 | 2 | 0 0 0 0 0] -> [2 2 2 | 2 | 3 0 0 3 0]`
#     - 若一行有两段 bar，例如近端长 2 颜色 4、远端长 3 颜色 6：
#       原 ext 仍按近端优先、后端不覆盖地打点；完成后 bar_side 上这两段都一起变成 separator 颜色。
#   complexity_calibration:
#     - recolor 规则固定为“变成分隔条颜色”，线索完全来自可见结构，不依赖颜色 id、排序或隐藏状态。
#     - 变体不会增加新的对象类别，只让输出比原题多一个稳定、可重复观察到的变化，复杂度与原题同量级。
#     - 输入仍保留原 bar 的真实颜色，因此解题者必须同时从输出中看出两件事：
#       1) ext_side 的周期落点规则；2) bar_side 会整体改色。
#   implicit_assumption_review:
#     - `_find_sep_col` 仍假设存在一列几乎整列、且非背景部分同色的竖直分隔条。生成器继续保证这点。
#     - `bars_on_left = left_mass >= right_mass` 假设输入的 bar 只在分隔条一侧出现。生成器仍只在单侧放 bar。
#     - `_iter_runs_1d` 把 bar 视为同色连续段；D5 不改变 bar 在输入中的几何结构，因此仍成立。
#     - 原 `main()` 对 `out` 只改 ext_side；这里新增“回写 bar_side”为 separator 颜色。这个写法依赖 bar
#       坐标来自输入而非输出，否则 recolor 后 bar 会彼此融合、无法再反推。实现中先从输入收集 bar，再统一回写。
#     - ext-side 打点必须先于 recolor 执行；若先 recolor，输出虽然同样可写出，但逻辑上会把“源颜色”概念弄混。
#   placement_legality:
#     - bar 仍只放在 separator 一侧，长度总和不超过 bar_side 宽度，双 bar 时保持相邻。
#     - ext_side 在输入中保持全背景，避免 solver 将额外非背景误认成 bar 或已占用像素。
#     - separator 仍宽 1 列、允许至多 1 个非 active terminal-row 缺口，保持与原题一致。
#   observability_and_cleanliness:
#     - 输出中的 ext_side 仍是稀疏周期点阵；bar_side 则变成与 separator 同色的水平短条，视觉差异明显。
#     - 虽然 gap=0 时 bar 会与 separator 融合成 T 形/加宽结构，但输入里原 bar 颜色明确，输出变化仍可直接观察。
#   color_change_inference:
#     - recolor 目标不是任意映射，而是结构上唯一显眼的分隔条颜色；2-3 组样例即可支持人类归纳。

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
        if int(row[c]) == int(bg):
            c += 1
            continue
        col = int(row[c])
        c0 = c
        while c < c_hi and int(row[c]) == col:
            c += 1
        yield c0, c - 1, col


def _collect_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> list[tuple[int, int, int, int, int, int]]:
    """Return bars as (dist_to_sep, row, c0, c1, length, color)."""
    h, w = g.shape
    bars: list[tuple[int, int, int, int, int, int]] = []
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


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)

    sep_col, sep_color = _find_sep_col(g, bg)
    if sep_col is None or sep_color is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    # New assumption introduced by D5:
    # all bar coordinates must be collected from the input before recoloring, because the recolor step
    # deliberately destroys the original bar colors and may merge adjacent segments visually.
    bars = _collect_bars(g, sep_col, bars_on_left, bg)

    # Original rule: paint the extension nearest-first without overwriting non-bg cells.
    for _, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if int(out[r, cc]) == bg:
                out[r, cc] = col

    # D5 recolor: once the extension is complete, every source bar becomes the separator color.
    for _, r, c0, c1, _length, _col in bars:
        out[r, c0 : c1 + 1] = sep_color

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
    grid: np.ndarray,
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
            grid[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            grid[row, cursor : cursor + length] = color
            cursor += length


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    """Simulate the original nearest-first periodic extension on the ext side."""
    w = out.shape[1]
    for length, color in bars:
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if int(out[row, cc]) == 0:
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
    active_rows_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_rows_set]

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        if gap_row == 0 or gap_row == h - 1:
            inp[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_specs: dict[int, tuple[list[tuple[int, int]], int]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        row_specs[row] = (bars, sep_gap)
        _place_bars(inp, row, sep_col, bars_on_left, bars, sep_gap)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None

    expected = inp.copy()
    for row, (bars, sep_gap) in row_specs.items():
        _paint_extension(expected, row, sep_col, bars_on_left, bars)
        recolored = [(length, sep_color) for length, _color in bars]
        _place_bars(expected, row, sep_col, bars_on_left, recolored, sep_gap)

    out = main(inp)
    if not np.array_equal(out, expected):
        return None

    # Variant guard: at least one bar pixel must recolor to the separator color in the output.
    bar_mask = (inp != 0) & (inp != sep_color)
    if not np.all(out[bar_mask] == sep_color):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
