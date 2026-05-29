# Combo variant of 1ae2feb7 (index 2)
# combo_spec: D4/0+D14/0
# Name: 宽涂射线遇障碍停止

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis
# ----------------
# a) 具体化与行为追踪
# - 保留原题框架：一列近乎贯穿全高的竖直 separator，把网格分成 bar_side 与 ext_side；
#   所有水平单色色条都在同一侧，ext_side 是它们向外延伸的区域。
# - 融合后 ext_side 输入中预置了若干同色单像素障碍。输出时仍按“离 separator 近的 bar 先处理，
#   已有颜色不覆盖”的顺序扩展，但每次周期事件不再只落一个点，而是落一整段与该 bar 等长的连续段。
# - 连续段不是整块写入，而是按射线方向逐格扫描：若遇到障碍色，则当前段立刻在障碍前截断，
#   且这条 bar 的后续周期事件全部停止；非障碍但已上色的格子只跳过，不会中断射线。
# - 小例子（bars 在左，separator 在列 4，bar 长度 3，障碍在 ext offset=1）：
#   输入行: 1112 0 8 0 0 0
#           ^^^ bar      ^ obstacle
#   原题 point/no-stop 会在 offset 0,3,... 落单点；D4(point/stop) 因障碍不在锚点上，仍会继续落点；
#   D14(dense/no-stop) 会把 ext_side 整段刷满；融合版 dense/stop 只会把第一格涂上，然后在第二格障碍处停止。
#   差异恰好要求解题者同时识别“宽涂”和“遇障碍终止”。
#
# b) 复杂度校准
# - D14 的“宽涂”如果放在完全空白的 ext_side，上色并集会天然变成连续带，容易让 bar 长度本身退到不可见。
#   因此生成器强制把障碍放在“不是周期锚点的位置”，使融合版相对 D4 单变体显著不同：
#   point/stop 会越过该障碍继续落点，而 dense/stop 会在段内被截断。
# - 同时拒绝融合版与 D14(dense/no-stop) 相同的样本，确保“障碍终止”也实际起作用。
# - 两条 bar 同行时，近侧 dense bar 往往会遮掉远侧 bar 在 ext_side 的贡献；这是融合后的自然结果。
#   为避免视觉噪声，生成器降低双 bar 频率，但不完全禁用，仍保留原题的对象形态。
#
# c) 对 main() 隐式前提的审查
# - `_find_sep_col` 只依赖 separator 列“非背景且几乎全高同色”，障碍落在 ext_side，不破坏该前提。
# - 原版用 `left_mass >= right_mass` 判 bar_side；障碍加入后 ext_side 也有非背景像素，会误判。
#   这里改成比较两侧“非背景颜色种类数”：bar_side 至少含多个 bar 颜色，ext_side 只有一种障碍色。
# - 原版扩展循环默认“非背景只代表更近 bar 已经上色，因此直接跳过即可”。融合后 ext_side 还会有障碍，
#   所以 dense 扫描必须把“障碍导致终止”和“已有颜色仅跳过”区分开。
# - D14 把“单点写入”改成“整段扫描”，所以每个周期事件都必须逐格处理，而不是整片切片赋值后再判断。
#
# d) 放置合法性
# - generator 采用 output-first：先摆 separator、bars、障碍，再调用统一模拟器得到融合版输出，
#   最后清空 ext_side 中的非障碍颜色形成输入。
# - 障碍只放在 ext_side，且每个活跃行恰放一个；不会与 bars、separator 重叠。
# - 障碍偏移只从 `1 .. ext_side_len-2` 里选，并要求 `offset % near_len != 0`，保证它不在 point 规则的锚点上，
#   同时其后仍留有至少一个 ext_side 格子，避免“贴边终止”过弱。
#
# e) 视觉可观察性
# - 输入中 ext_side 除了障碍外全空，因此同色单像素很容易被识别成特殊标记。
# - 输出中融合版是“从 separator 开始的连续前缀 + 障碍保留 + 障碍后留白”，与两个单变体都不同：
#   对 D4 来说它不是稀疏点列，对 D14 来说它不会一路刷到边缘。
# - 障碍列在不同行变化，避免被误读成固定模板列。
#
# f) 可视化整洁度
# - 每行最多一个障碍，且活跃行之间至少隔一空行，避免多行信息缠在一起。
# - 双 bar 仍只存在于 bar_side，ext_side 上没有多层颜色混叠；唯一保留的非输出标记只有障碍色。
#
# g) 变化的可推断性
# - 没有颜色映射。bar 颜色直接决定输出颜色，障碍色只负责“停止”语义。
# - 新 solver 的额外假设是：输入 ext_side 至多出现一种非背景颜色（障碍色），且 bar_side 总体颜色种类数
#   严格多于 ext_side。generator 用“活跃行主色互异 + 障碍色单一且不出现在 bar_side”来保证这一点。


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


def _paint_dense_segment(
    out: np.ndarray,
    row: int,
    start: int,
    end: int,
    step: int,
    color: int,
    bg: int,
    obstacle_color: int | None,
    stop_at_obstacle: bool,
) -> bool:
    """Paint one dense segment in scan order; return True iff an obstacle stopped the ray."""
    for cc in range(start, end, step):
        cell = int(out[row, cc])
        if stop_at_obstacle and obstacle_color is not None and cell == obstacle_color:
            return True
        if cell == bg:
            out[row, cc] = color
    return False


def _paint_bar_ray(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    length: int,
    color: int,
    bg: int,
    obstacle_color: int | None,
    *,
    dense: bool,
    stop_at_obstacle: bool,
) -> None:
    w = out.shape[1]
    if length <= 0:
        return

    if dense:
        if bars_on_left:
            for start in range(sep_col + 1, w, length):
                end = min(w, start + length)
                if _paint_dense_segment(
                    out,
                    row,
                    start,
                    end,
                    1,
                    color,
                    bg,
                    obstacle_color,
                    stop_at_obstacle,
                ):
                    break
        else:
            for anchor in range(sep_col - 1, -1, -length):
                start = anchor
                end = max(-1, anchor - length)
                if _paint_dense_segment(
                    out,
                    row,
                    start,
                    end,
                    -1,
                    color,
                    bg,
                    obstacle_color,
                    stop_at_obstacle,
                ):
                    break
        return

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
    dense: bool,
    stop_at_obstacle: bool,
) -> np.ndarray:
    g = np.asarray(g, dtype=int)
    bg = _bg_color(g)
    h, w = g.shape

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
        for length, col in row_bars:
            _paint_bar_ray(
                out,
                r,
                sep_col,
                bars_on_left,
                length,
                col,
                bg,
                obstacle_color,
                dense=dense,
                stop_at_obstacle=stop_at_obstacle,
            )
    return out


# Overview:
# Solve a fused 1ae2feb7 variant where each periodic paint event writes a full-length segment,
# and each bar's ray stops permanently when that segment first encounters an obstacle pixel.
#
# Input structure:
# - One near-solid vertical separator column splits the grid into a bar_side and an ext_side.
# - Active rows on the bar_side contain 1-2 horizontal monochrome bars, ordered by proximity to the separator.
# - The ext_side is otherwise blank except for a few single-pixel obstacles, all sharing one color.
#
# Rule:
# - For each row, process bars from nearest to farthest exactly as in the base task.
# - A bar of length L no longer paints a single periodic point; at each periodic anchor it paints an L-cell segment
#   extending away from the separator, skipping already-colored cells but never overwriting them.
# - If the scan of any such segment reaches an obstacle pixel, that segment is truncated before the obstacle and the
#   same bar performs no later periodic events. Obstacles stay unchanged.
#
# Key insight:
# - The obstacle must be interpreted as a hard stop encountered during dense segment painting, not merely as a cell to
#   skip. This is why off-anchor obstacles cut off a solid prefix in the fused output while the single-variant rules do not.
def main(grid: np.ndarray) -> np.ndarray:
    return _simulate(np.asarray(grid, dtype=int), dense=True, stop_at_obstacle=True)


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 5, 14

PALETTE_MIN, PALETTE_MAX = 5, 7

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.25
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 500


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
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity < 2:
        raise ValueError("capacity must allow a near bar of length >= 2")

    two_bars = capacity >= 3 and rng.random() < P_TWO_BARS
    if not two_bars:
        near_len = int(rng.integers(2, capacity + 1))
        return [(near_len, primary)], sep_gap

    near_len = int(rng.integers(2, capacity))
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
    cols = []
    for offset in range(1, ext_side_len - 1):
        if offset % near_len == 0:
            continue
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

    for _ in range(30):
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
    ext = inp[:, sep_col + 1 :] if bars_on_left else inp[:, :sep_col]
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

    base = np.zeros((h, w), dtype=int)
    base[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        base[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len)
        row_bars_map[row] = bars
        _place_bars(base, row, sep_col, bars_on_left, bars, sep_gap)

    obstacle_cols = _sample_obstacle_cols(rng, active_rows, row_bars_map, sep_col, bars_on_left, ext_side_len)
    if obstacle_cols is None:
        return None
    for row, cc in obstacle_cols.items():
        base[row, cc] = obstacle_color

    combo_out = _simulate(base, dense=True, stop_at_obstacle=True)
    point_stop_out = _simulate(base, dense=False, stop_at_obstacle=True)
    dense_nostop_out = _simulate(base, dense=True, stop_at_obstacle=False)
    point_nostop_out = _simulate(base, dense=False, stop_at_obstacle=False)

    combo_vs_point_stop = []
    combo_vs_dense_nostop = []
    combo_vs_point_nostop = []
    for row in active_rows:
        combo_vs_point_stop.append(int(np.count_nonzero(combo_out[row, :] != point_stop_out[row, :])))
        combo_vs_dense_nostop.append(int(np.count_nonzero(combo_out[row, :] != dense_nostop_out[row, :])))
        combo_vs_point_nostop.append(int(np.count_nonzero(combo_out[row, :] != point_nostop_out[row, :])))

    if min(combo_vs_point_stop) <= 0:
        return None
    if min(combo_vs_dense_nostop) <= 0:
        return None
    if min(combo_vs_point_nostop) <= 0:
        return None

    inp = _clear_ext_side_keep_obstacles(combo_out, sep_col, bars_on_left, obstacle_color)

    if get_bg(inp) != 0:
        return None
    if obstacle_color not in colors_present(inp, exclude=[0]):
        return None

    left_colors = _side_colors(inp, 0, sep_col, 0)
    right_colors = _side_colors(inp, sep_col + 1, w, 0)
    if min(len(left_colors), len(right_colors)) != 1:
        return None
    if max(len(left_colors), len(right_colors)) < 2:
        return None

    if not np.array_equal(main(inp), combo_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
