# Combo variant of 1ae2feb7 (index 29)
# combo_spec: D4/0+D7/3+D11/0
# Name: 延迟起始覆盖模式障碍终止

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 保留原题的主体对象：一列几乎贯穿全高的竖直分隔条、同侧的水平单色色条、另一侧作为扩展区。
# - 融合后，输入在扩展区额外出现单像素障碍物，且扩展规则同时改三点：
#   1. 每条 bar 的周期点列不再从紧邻 separator 的格子开始，而是从“距 separator 恰好一个 bar 长度”的位置开始。
#   2. 同一行仍按 near->far 的几何顺序处理 bars，但后处理的远条可以覆盖先处理的近条颜色。
#   3. 某条 ray 只要下一次目标位置正好是障碍物，就立刻停止，障碍物本身保持不变。
# - 小例子（bars 在左，separator 在列 4）：
#   输入行: 1 3 3 2 0 0 0 8 0 0 0
#           ^ near 长度=1，far 长度=2，8 为障碍
#   原题：near 从列 5 起涂 5,6,7,8...，far 从列 5 起涂 5,7,9...，且不覆盖；几乎紧贴 separator 就着色。
#   融合后：near 只尝试列 6,7,8...；far 只尝试列 7,9,...。near 先把列 6、7 涂成其颜色，far 再把列 7 覆盖成远条颜色；
#   当两条 bar 都下一次命中列 8 的障碍时立即停止，不再继续到更远位置。差异正是“起点后移 + 远条覆盖 + 障碍截断”。
#
# b) 复杂度校准
# - 仍然只有 1-2 条 bar/活跃行，避免引入过多层级；障碍物也只有一种颜色。
# - generator 强制至少一行出现真实覆盖，且每个活跃行的最近 bar 都会被障碍截断，这样 2-3 个样例就能同时观察到三项变化。
# - 为了让 D11 可见，所有活跃行都保证至少存在一个从 length 偏移开始的可达落点；为让 D4 可见，还保证障碍后面本来还存在后续落点。
#
# c) 对原 main() 隐式假设的逐行审查
# - `_find_sep_col` 只依赖“separator 列几乎整列同色”，障碍不破坏这一点。
# - 原版用左右两侧非背景像素数判断哪侧是 bar_side；融合后 ext_side 有障碍物，短 bar 场景会误判，所以改成优先比较颜色种类数：
#   bar_side 至少有多种 bar 颜色，而 ext_side 只有一种障碍色。
# - 原版扩展循环默认“非背景意味着已有正确输出，不能覆盖”；D7/3 直接推翻这个前提，因此扩展时必须允许 later write 覆盖 earlier write。
# - 原版扩展从 separator 邻格起步；D11/0 推翻这个前提，因此所有位置枚举都要从 `length` 偏移开始。
# - 原版 ext_side 没有静态物体；D4/0 推翻这个前提，因此每次落点都要先判障碍，再决定是否写入或停止。
#
# d) 放置合法性
# - bars 仍只放在 separator 同一侧，且按 near->far 相邻堆叠，不重叠不越界。
# - 障碍只放在 ext_side，且放在某个真实周期落点上；每个活跃行都保证该障碍后面原本还有至少一个可达位置，因此“停止”一定改变结果。
# - 输出仍采用 output-first 生成：先摆 separator、bars、障碍，再跑统一模拟器得到输出，最后清空非障碍扩展区得到输入。
#
# e) 视觉可观察性
# - 所有活跃行在 separator 旁边都会留出一段起始空白，这是 D11 的稳定线索。
# - 至少一行使用两条不同颜色的 bars，并强制它们在障碍前先发生一次碰撞，因此能肉眼看出“远条颜色最终保留”。
# - 障碍统一颜色且为单像素，位置随行变化，容易被识别为停止标记而不是随机噪声。
#
# f) 可视化整洁度
# - 行与行之间仍保留空隔；覆盖只发生在单行内部，不会产生跨行缠绕。
# - 同一区域的叠加层次只有两层：bar 颜色的覆盖，以及不可覆盖的障碍物，视觉上仍清楚。
#
# g) 变化的可推断性
# - 没有颜色映射：输出颜色始终直接来自同一行的 bars，障碍色只承担“终止边界”角色。
# - 组合规律依赖几何关系而非颜色值大小、人类不可见排序或随机 tie-break。
#
# 新隐式假设审查
# - solver 现在假设 ext_side 输入中至多存在一种非背景颜色，并把它视为障碍色；generator 保证这一点。
# - solver 假设 bar_side 的颜色种类数严格多于 ext_side；generator 通过至少两个不同的主 bar 颜色保证这一点。
# - solver 假设每个活跃行的 bars 在同一侧且是一维连续段；generator 仍按原题结构构造。
# - solver 允许单行内 later overwrite earlier；generator 强制至少一行真正触发这一覆盖差异，避免退化为“看起来像 D4+D11”。


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


def _paint_extension_row(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    bg: int,
    obstacle_color: int | None,
    *,
    start_multiplier: int,
    overwrite: bool,
    stop_at_obstacle: bool,
) -> None:
    w = out.shape[1]
    for length, color in bars:
        if length <= 0:
            continue
        if bars_on_left:
            cc = sep_col + 1 + start_multiplier * length
            delta = length
        else:
            cc = sep_col - 1 - start_multiplier * length
            delta = -length
        while 0 <= cc < w:
            cell = int(out[row, cc])
            if stop_at_obstacle and obstacle_color is not None and cell == obstacle_color:
                break
            if overwrite or cell == bg:
                out[row, cc] = color
            cc += delta


def _simulate(
    g: np.ndarray,
    *,
    start_multiplier: int,
    overwrite: bool,
    stop_at_obstacle: bool,
) -> np.ndarray:
    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    bars_on_left = _bars_side_is_left(g, sep_col, bg)
    ext_colors = _side_colors(g, sep_col + 1, g.shape[1], bg) if bars_on_left else _side_colors(g, 0, sep_col, bg)
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
            start_multiplier=start_multiplier,
            overwrite=overwrite,
            stop_at_obstacle=stop_at_obstacle,
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the fused periodic-extension rule with delayed start, overwrite, and obstacle stop.

    Input structure:
    - One nearly solid vertical separator column splits the grid into a bar side and an extension side.
    - Active rows contain 1-2 horizontal monochrome bars on the same side of the separator.
    - The extension side is otherwise empty except for single-pixel obstacles of one shared color.

    Rule:
    - Process each row's bars from nearest to farthest relative to the separator.
    - For a bar of length L, start painting at distance L from the separator, then continue every L cells.
    - Later bars overwrite earlier painted cells, but if a target cell is an obstacle, that bar stops immediately and the obstacle remains unchanged.

    Key insight:
    - The correct output is determined by the interaction of three cues: the initial blank gap next to the separator, collisions where farther bars win, and fixed obstacle pixels that terminate the periodic rays.
    """
    return _simulate(
        np.asarray(grid, dtype=int),
        start_multiplier=1,
        overwrite=True,
        stop_at_obstacle=True,
    )


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 7, 14

PALETTE_MIN, PALETTE_MAX = 5, 7

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.5
P_SEP_GAP = 0.15
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


def _sample_regular_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    ext_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("bar capacity must stay positive")

    max_near_len = min(capacity, max(1, (ext_side_len - 1) // 3))
    if max_near_len <= 0:
        raise ValueError("nearest bar must admit a visible obstacle stop")

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
    far_color = int(rng.choice(others))
    return [(near_len, primary), (far_len, far_color)], sep_gap


def _forced_combo_row(
    palette: list[int],
    primary: int,
    row_index: int,
) -> list[tuple[int, int]]:
    others = [c for c in palette if c != primary]
    far_color = others[row_index % len(others)]
    return [(1, primary), (2, int(far_color))]


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


def _offset_to_col(sep_col: int, bars_on_left: bool, offset: int) -> int:
    return sep_col + 1 + offset if bars_on_left else sep_col - 1 - offset


def _row_obstacle_offset(
    ext_side_len: int,
    bars: list[tuple[int, int]],
    *,
    force_combo: bool,
) -> int | None:
    near_len = bars[0][0]
    max_offset = ext_side_len - 1
    if force_combo:
        return 4 if max_offset >= 4 else None

    max_k = max_offset // near_len
    candidates = [k * near_len for k in range(2, max_k) if (k + 1) * near_len <= max_offset]
    if not candidates:
        return None
    return int(candidates[0])


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


def _behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    combo = main(inp)
    no_stop = _simulate(inp, start_multiplier=1, overwrite=True, stop_at_obstacle=False)
    no_overwrite = _simulate(inp, start_multiplier=1, overwrite=False, stop_at_obstacle=True)
    no_delay = _simulate(inp, start_multiplier=0, overwrite=True, stop_at_obstacle=True)

    return {
        "combo_vs_no_stop": int(np.sum(combo != no_stop)),
        "combo_vs_no_overwrite": int(np.sum(combo != no_overwrite)),
        "combo_vs_no_delay": int(np.sum(combo != no_delay)),
    }


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
    if ext_side_len < 7:
        return None

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

    forced_row = int(rng.choice(active_rows))
    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    row_gap_map: dict[int, int] = {}
    row_obstacles: dict[int, int] = {}

    for idx, row in enumerate(active_rows):
        if row == forced_row:
            sep_gap = 1 if rng.random() < P_SEP_GAP else 0
            capacity = bar_side_len - sep_gap
            if capacity < 3:
                return None
            bars = _forced_combo_row(palette, row_primary[idx], idx)
        else:
            bars, sep_gap = _sample_regular_row_bars(
                rng,
                palette,
                row_primary[idx],
                bar_side_len,
                ext_side_len,
            )
        row_bars_map[row] = bars
        row_gap_map[row] = sep_gap
        _place_bars(base, row, sep_col, bars_on_left, bars, sep_gap)

    for row in active_rows:
        offset = _row_obstacle_offset(
            ext_side_len,
            row_bars_map[row],
            force_combo=(row == forced_row),
        )
        if offset is None:
            return None
        row_obstacles[row] = _offset_to_col(sep_col, bars_on_left, offset)
        base[row, row_obstacles[row]] = obstacle_color

    combo_out = main(base)
    inp = _clear_ext_side_keep_obstacles(combo_out, sep_col, bars_on_left, obstacle_color)

    if get_bg(inp) != 0:
        return None
    if obstacle_color not in colors_present(inp, exclude=[0]):
        return None

    left_colors = _side_colors(inp, 0, sep_col, 0)
    right_colors = _side_colors(inp, sep_col + 1, w, 0)
    if min(len(left_colors), len(right_colors)) != 1 or max(len(left_colors), len(right_colors)) < 2:
        return None

    if not np.array_equal(main(inp), combo_out):
        return None

    metrics = _behavior_metrics(inp)
    if min(metrics.values()) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
