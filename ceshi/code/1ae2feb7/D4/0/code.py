# Variant of 1ae2feb7
# Dimension: D4 — 终止条件/目标 (dst)
# Change: null（网格边缘）——射线延伸直到到达网格的对侧边缘，没有其他终止条件... → 遇到障碍物停止

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis
# ----------------
# a) 具体化与行为追踪
# - 保留原题主体：一整列竖直分隔条，把网格分成 bar_side 与 ext_side；bar_side 上每个活跃行有 1-2 条水平单色色条。
# - 变体只改 D4：ext_side 在输入中不再全空，而是出现若干“障碍物”单像素，全部使用同一种新颜色。
# - 输出时，仍按“离分隔条近的 bar 先处理、已有颜色不覆盖”的原顺序在同一行做周期点列扩展；
#   但某条 bar 的下一次落点如果正好是障碍物，就立即停止这条 bar 的后续扩展。障碍物本身保留不变。
# - 小例子（bars 在左，sep 在第 4 列）：
#   输入行: 1102 0 0 8 0 0
#           ^^ bar 长度=2，8 是障碍
#   原题会在 ext_side 的列 5,7,9,... 位置落 1；变体中会先落到列 5，再尝试列 7 时撞到 8，于是停止，
#   列 9 之后都不再着色。差异点正是“原先继续到边缘，现在在障碍前终止”。
#
# b) 复杂度校准
# - 不增加额外排序维度，不引入多类障碍或颜色映射。障碍只有一种颜色，视觉上很稳定。
# - 让不同活跃行的障碍位置变化，并且每个活跃行都真的截断至少一个本可继续出现的周期点；
#   这样 2-3 个样例就能看出“同色单像素是停止标记”，不会退化成普通稀疏噪声。
#
# c) 对原 main() 的逐行审查
# - `_find_sep_col` 依赖“separator 列的非背景格同色且几乎贯穿全高”，障碍不影响这一点，仍成立。
# - 原版 `left_mass >= right_mass` 依赖 ext_side 全空。加入障碍后，ext_side 也有非背景像素，短 bar 时会误判 sides。
#   这里改成比较 separator 两侧的“非背景颜色种类数”：bar_side 至少含多个 bar 颜色，ext_side 只有障碍这一种颜色。
# - 原版扩展循环默认 ext_side 非背景只会来自更近 bar 的已有着色，所以“非背景则跳过”是安全的。
#   变体中 ext_side 还会出现不可覆盖的障碍；因此必须先判障碍，再决定是停止还是跳过。
#
# d) 放置合法性
# - generator 仍是 output-first：先摆 separator、bars、障碍，再模拟带停止的扩展，最后把非障碍 ext_side 清空得到输入。
# - 障碍只放在 ext_side，不与 separator 或 bars 重叠；每个障碍都选在某个 bar 的真实周期落点上，
#   且不是该 bar 在该行的最后一个落点，这样“停止”一定会改变输出。
#
# e) 视觉可观察性
# - 障碍统一颜色、统一为单像素，且 ext_side 其余输入位置全空，解题者容易把它们识别为特殊标记而非普通输出残留。
# - 不同行障碍位置变化，避免误读成“固定列模板”。
#
# f) 可视化整洁度
# - 每行最多一个障碍，障碍与周期点列共存但不叠色；没有多层覆盖。
# - 保留原题的行间空隔，避免多行输出互相缠绕。
#
# g) 颜色变化可推断性
# - 障碍颜色不参与任何颜色映射，只承担“停止标记”这一结构角色；bar 颜色仍直接决定输出颜色。
#
# 额外生成约束 / 新隐式假设审查
# - solver 现在假设 ext_side 在输入中至多有一种非背景颜色，即障碍色；generator 保证这一点。
# - solver 用“bar_side 的颜色种类数 > ext_side 的颜色种类数”来判定方向；generator 保证活跃行 >= 2 且主 bar 颜色互异，
#   所以 bar_side 至少有 2 种颜色，而 ext_side 只有 1 种障碍色。
# - solver 假设障碍色不会出现在 bar_side；generator 保证障碍色与 separator / palette 全部互斥。


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
        if int(row[c]) == bg:
            c += 1
            continue
        col = int(row[c])
        c0 = c
        while c < c_hi and int(row[c]) == col:
            c += 1
        yield c0, c - 1, col


def _side_colors(g: np.ndarray, c_lo: int, c_hi: int, bg: int) -> list[int]:
    if c_lo >= c_hi:
        return []
    return colors_present(g[:, c_lo:c_hi], exclude=[bg])


def _bars_side_is_left(g: np.ndarray, sep_col: int, bg: int) -> bool:
    left_colors = _side_colors(g, 0, sep_col, bg)
    right_colors = _side_colors(g, sep_col + 1, g.shape[1], bg)
    if len(left_colors) != len(right_colors):
        return len(left_colors) > len(right_colors)

    # Fallback for malformed inputs: prefer the side with more non-bg mass.
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    return left_mass >= right_mass


def _collect_bars(g: np.ndarray, sep_col: int, bars_on_left: bool, bg: int):
    bars = []
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


def _paint_extension_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    bg: int,
    obstacle_color: int | None,
    *,
    stop_at_obstacle: bool,
) -> None:
    w = out.shape[1]
    for length, color in bars:
        if length <= 0:
            continue
        if bars_on_left:
            cc = sep_col + 1
            while cc < w:
                cell = int(out[row, cc])
                if stop_at_obstacle and obstacle_color is not None and cell == obstacle_color:
                    break
                if cell == bg:
                    out[row, cc] = color
                cc += length
        else:
            cc = sep_col - 1
            while cc >= 0:
                cell = int(out[row, cc])
                if stop_at_obstacle and obstacle_color is not None and cell == obstacle_color:
                    break
                if cell == bg:
                    out[row, cc] = color
                cc -= length


def _simulate(
    g: np.ndarray,
    *,
    stop_at_obstacle: bool,
) -> np.ndarray:
    h, w = g.shape
    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    bars_on_left = _bars_side_is_left(g, sep_col, bg)
    ext_colors = _side_colors(g, sep_col + 1, w, bg) if bars_on_left else _side_colors(g, 0, sep_col, bg)
    obstacle_color = ext_colors[0] if len(ext_colors) == 1 else None

    bars_full = _collect_bars(g, sep_col, bars_on_left, bg)
    per_row: dict[int, list[tuple[int, int]]] = {}
    for _, r, _c0, _c1, length, col in bars_full:
        per_row.setdefault(r, []).append((length, col))

    out = g.copy()
    for r, row_bars in per_row.items():
        _paint_extension_row(
            out,
            r,
            sep_col,
            bars_on_left,
            row_bars,
            bg,
            obstacle_color,
            stop_at_obstacle=stop_at_obstacle,
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _simulate(np.asarray(grid, dtype=int), stop_at_obstacle=True)


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 5, 14

PALETTE_MIN, PALETTE_MAX = 5, 7

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.45
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(
    rng: np.random.Generator,
    active_count: int,
) -> tuple[int, int, list[int]]:
    colors = list(range(1, 10))
    sep_color, obstacle_color = map(int, rng.choice(colors, size=2, replace=False))
    bar_pool = [c for c in colors if c not in {sep_color, obstacle_color}]
    size_lo = max(PALETTE_MIN, active_count)
    size_hi = min(PALETTE_MAX, len(bar_pool))
    size = int(rng.integers(size_lo, size_hi + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, obstacle_color, palette


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    ext_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("bar capacity must be positive")

    # Ensure the nearest bar has at least one interior obstacle candidate and at least one
    # would-be paint position beyond that obstacle.
    max_near_len = max(1, (ext_side_len - 1) // 2)
    max_near_len = min(max_near_len, capacity)

    two_bars = capacity >= 2 and rng.random() < P_TWO_BARS
    if not two_bars:
        length = int(rng.integers(1, max_near_len + 1))
        return [(length, primary)], sep_gap

    near_len_hi = min(max_near_len, capacity - 1)
    if near_len_hi <= 0:
        length = int(rng.integers(1, max_near_len + 1))
        return [(length, primary)], sep_gap

    near_len = int(rng.integers(1, near_len_hi + 1))
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


def _candidate_obstacle_cols(
    sep_col: int,
    bars_on_left: bool,
    ext_side_len: int,
    near_len: int,
) -> list[int]:
    max_k = (ext_side_len - 1) // near_len
    if max_k < 2:
        return []
    cols = []
    for k in range(1, max_k):
        offset = k * near_len
        cc = sep_col + 1 + offset if bars_on_left else sep_col - 1 - offset
        cols.append(int(cc))
    return cols


def _sample_obstacle_cols(
    rng: np.random.Generator,
    active_rows: list[int],
    row_bars_map: dict[int, list[tuple[int, int]]],
    sep_col: int,
    bars_on_left: bool,
    ext_side_len: int,
) -> dict[int, int] | None:
    row_candidates: dict[int, list[int]] = {}
    for row in active_rows:
        near_len = row_bars_map[row][0][0]
        candidates = _candidate_obstacle_cols(sep_col, bars_on_left, ext_side_len, near_len)
        if not candidates:
            return None
        row_candidates[row] = candidates

    for _ in range(20):
        choice = {row: int(rng.choice(cands)) for row, cands in row_candidates.items()}
        if len(set(choice.values())) >= min(2, len(active_rows)):
            return choice
    return None


def _clear_ext_side_keep_obstacles(
    out: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    obstacle_color: int,
) -> np.ndarray:
    inp = out.copy()
    if bars_on_left:
        ext = inp[:, sep_col + 1 :]
    else:
        ext = inp[:, :sep_col]
    ext[ext != obstacle_color] = 0
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
    ext_side_len = w - 1 - bar_side_len

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    sep_color, obstacle_color, palette = _sample_palette(rng, active_count)

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len, ext_side_len)
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    obstacle_cols = _sample_obstacle_cols(rng, active_rows, row_bars_map, sep_col, bars_on_left, ext_side_len)
    if obstacle_cols is None:
        return None
    for row, cc in obstacle_cols.items():
        out[row, cc] = obstacle_color

    out_stop = _simulate(out, stop_at_obstacle=True)
    out_no_stop = _simulate(out, stop_at_obstacle=False)

    blocked_rows = [
        row
        for row in active_rows
        if not np.array_equal(out_stop[row, :], out_no_stop[row, :])
    ]
    if len(blocked_rows) != len(active_rows):
        return None

    inp = _clear_ext_side_keep_obstacles(out_stop, sep_col, bars_on_left, obstacle_color)

    if get_bg(inp) != 0:
        return None
    if obstacle_color not in colors_present(inp, exclude=[0]):
        return None

    left_colors = _side_colors(inp, 0, sep_col, 0)
    right_colors = _side_colors(inp, sep_col + 1, w, 0)
    if min(len(left_colors), len(right_colors)) != 1 or max(len(left_colors), len(right_colors)) < 2:
        return None

    if not np.array_equal(main(inp), out_stop):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
