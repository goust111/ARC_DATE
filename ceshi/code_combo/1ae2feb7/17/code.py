# Combo variant of 1ae2feb7 (index 17)
# combo_spec: D4/0+D7/3
# Name: 障碍终止与后到先得覆盖

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 保留原题骨架：一列几乎贯穿全高的竖直分隔条，把网格分成 bar_side 与 ext_side；bar_side 的活跃行上有 1-2 条水平单色色条。
# - 融合后，输入的 ext_side 不再是纯空，而是保留若干同色单像素障碍物。输出仍按“离 separator 近到远”的顺序处理每行 bars。
# - 变化 1（D4/0）：某条 bar 的周期射线如果下一次正好落到障碍物，就在该障碍处停止，障碍本身保持原色且不可覆盖。
# - 变化 2（D7/3）：取消旧题的 no-overwrite。近 bar 先涂，远 bar 后涂；如果两条 bar 落到同一非障碍格，后处理的远 bar 覆盖前者。
# - 小例子（bars 在左，separator 在第 4 列，ext 向右）：
#     输入行: 3 3 5 5 2 0 0 8 0 0 0
#             near=55(len=2), far=33(len=2), 8 是障碍
#   原题：第 5,7,9... 列先被 near 涂，far 因 no-overwrite 无法改写，且不会在障碍处停止。
#   融合后：near 先涂第 5,7；下一步命中障碍列 8 后停止。far 再从第 5 开始，能覆盖第 5,7 的 near 颜色，
#   但遇到障碍列 8 时同样停止，且不能把 8 改色。差异点是“后到先得覆盖”和“障碍不可覆盖且会截断”同时生效。
#
# b) 复杂度校准
# - 仍只保留一种障碍颜色，避免把“停止标记”做成复杂类型系统。
# - 每个样例至少有一条双 bar 行，这样 overwrite 语义一定可见；同时每条活跃行都放障碍并要求最终输出受其影响，
#   避免障碍只是装饰噪声。
# - 不引入额外排序键；处理顺序仍是纯几何的“距 separator 近到远”。
#
# c) 对原 main() 的隐式假设审查
# - `_find_sep_col` 假设 separator 列的非背景格几乎全列同色。障碍只出现在 ext_side，不破坏该条件。
# - 原版 `left_mass >= right_mass` 假设 ext_side 在输入中全空；加入障碍后该前提失效，可能把障碍侧误判成 bar_side。
#   这里改成比较两侧的“非背景颜色种类数”：bar_side 至少有多个 bar 颜色，而 ext_side 只有一种障碍色。
# - 原版扩展时 `if out[r, cc] == bg` 默认“非背景只来自更近 bar 的既有涂点”。融合后 ext_side 还可能是障碍；
#   因此扩展必须先检测障碍并停止，再决定是否写入或覆盖。
# - 允许覆盖后，不能再把“已有非背景就跳过”当作不变量；只有障碍仍是绝对不可写的格子。
#
# d) 放置合法性
# - generator 采用 output-first：先放 separator、bars、障碍，再用融合规则模拟输出，最后清掉 ext_side 的普通涂点得到输入。
# - 障碍只放在 ext_side，且位置必须是至少一条 bar 的真实周期落点，同时不是该 bar 的最后一个落点，确保“停止”确实会删掉后续本可出现的点。
# - bars 仍只在 separator 一侧、同行相邻堆叠，不与 separator 或障碍重叠。
#
# e) 视觉可观察性
# - 双 bar 行的第一个 ext 格一定是冲突点：旧题保留近 bar 颜色，融合版保留远 bar 颜色，人眼容易看出覆盖语义反转。
# - 障碍统一颜色、统一单像素，且 ext_side 的其余输入格为空，因此容易被识别为“特殊停止标记”。
# - 生成器要求至少一行同时体现“被障碍截断”和“后 bar 覆盖前 bar”，避免两个改动分别散落在不同样例角落里。
#
# f) 可视化整洁度
# - 活跃行之间保持至少一空行，所有相互作用都局限在单行内，不会出现跨行缠绕。
# - 每行只保留少量障碍单像素和稀疏周期点；尽管允许覆盖，但最终可见层次仍只有 bars / 障碍 / 输出点三类。
#
# g) 变化的可推断性
# - 没有颜色映射；输出点颜色直接继承本行已有 bar 颜色，障碍颜色始终只承担“停止且不可覆盖”的结构角色。
# - 生成器保证 bar 颜色与障碍色、separator 色互斥，并保证 bar_side 在输入中至少出现两种 bar 颜色，从而支撑方向判断与人类归纳。
#
# 新 solver 假设与 generator 对应保证
# - solver 假设输入 ext_side 恰好只有一种非背景色，即障碍色；generator 只在 ext_side 保留该单一颜色。
# - solver 用“颜色种类数更多的一侧是 bar_side”；generator 让活跃行主色互异，因此 bar_side 至少有两种 bar 颜色。
# - solver 把 ext_side 的唯一非背景色视为障碍；generator 保证该颜色不会出现在 bar_side 或 separator 上。


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
    stop_at_obstacle: bool,
    overwrite: bool,
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
                if overwrite or cell == bg:
                    out[row, cc] = color
                cc += length
        else:
            cc = sep_col - 1
            while cc >= 0:
                cell = int(out[row, cc])
                if stop_at_obstacle and obstacle_color is not None and cell == obstacle_color:
                    break
                if overwrite or cell == bg:
                    out[row, cc] = color
                cc -= length


def _simulate(
    g: np.ndarray,
    *,
    stop_at_obstacle: bool,
    overwrite: bool,
) -> np.ndarray:
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
    for _dist, r, _c0, _c1, length, col in bars_full:
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
            overwrite=overwrite,
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the extension side using periodic row projections with obstacle stopping and later-bar overwrite.

    Input structure:
    - A mostly solid vertical separator column splits the grid into a bar side and an extension side.
    - Active rows on the bar side contain 1-2 horizontal monochrome bars ordered by distance to the separator.
    - The extension side is otherwise empty except for single-pixel obstacles, all sharing one color.

    Rule:
    - For each row, process bars from nearest to farthest from the separator.
    - Each bar projects its color periodically onto the extension side with step = bar length, starting at the cell adjacent to the separator.
    - If a projected landing cell is an obstacle, that bar stops immediately and the obstacle remains unchanged.
    - On non-obstacle cells, later bars overwrite earlier projected dots.

    Key insight:
    - The bar order is still geometric near-to-far, but only obstacles are protected; ordinary projected dots are provisional and can be replaced by later bars.
    """
    return _simulate(np.asarray(grid, dtype=int), stop_at_obstacle=True, overwrite=True)


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 5, 14

PALETTE_MIN, PALETTE_MAX = 5, 7

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_TWO_BARS = 0.55
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
    ext_side_len: int,
    *,
    force_two: bool,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("bar capacity must be positive")

    max_len_with_obstacle = min(capacity, max(1, (ext_side_len - 1) // 2))
    two_bars = force_two or (capacity >= 2 and rng.random() < P_TWO_BARS)

    if not two_bars or capacity == 1:
        length = int(rng.integers(1, max_len_with_obstacle + 1))
        return [(length, primary)], sep_gap

    near_len_hi = min(max_len_with_obstacle, capacity - 1)
    if near_len_hi <= 0:
        length = int(rng.integers(1, max_len_with_obstacle + 1))
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


def _candidate_obstacle_cols_for_length(
    sep_col: int,
    bars_on_left: bool,
    ext_side_len: int,
    length: int,
) -> list[int]:
    max_k = (ext_side_len - 1) // length
    if max_k < 2:
        return []
    out = []
    for k in range(1, max_k):
        offset = k * length
        cc = sep_col + 1 + offset if bars_on_left else sep_col - 1 - offset
        out.append(int(cc))
    return out


def _row_obstacle_candidates(
    sep_col: int,
    bars_on_left: bool,
    ext_side_len: int,
    bars: list[tuple[int, int]],
) -> list[int]:
    candidates: set[int] = set()
    for length, _color in bars:
        candidates.update(_candidate_obstacle_cols_for_length(sep_col, bars_on_left, ext_side_len, length))
    return sorted(candidates)


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
        candidates = _row_obstacle_candidates(sep_col, bars_on_left, ext_side_len, row_bars_map[row])
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

    fusion_row = int(rng.choice(active_rows))
    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            ext_side_len,
            force_two=(row == fusion_row),
        )
        row_bars_map[row] = bars
        _place_bars(base, row, sep_col, bars_on_left, bars, sep_gap)

    obstacle_cols = _sample_obstacle_cols(rng, active_rows, row_bars_map, sep_col, bars_on_left, ext_side_len)
    if obstacle_cols is None:
        return None
    for row, cc in obstacle_cols.items():
        base[row, cc] = obstacle_color

    out_combo = _simulate(base, stop_at_obstacle=True, overwrite=True)
    out_stop_no_overwrite = _simulate(base, stop_at_obstacle=True, overwrite=False)
    out_no_stop_overwrite = _simulate(base, stop_at_obstacle=False, overwrite=True)
    out_orig = _simulate(base, stop_at_obstacle=False, overwrite=False)

    blocked_rows = [
        row
        for row in active_rows
        if not np.array_equal(out_combo[row, :], out_no_stop_overwrite[row, :])
    ]
    if len(blocked_rows) != len(active_rows):
        return None

    overwritten_rows = [
        row
        for row in active_rows
        if not np.array_equal(out_combo[row, :], out_stop_no_overwrite[row, :])
    ]
    if fusion_row not in blocked_rows or fusion_row not in overwritten_rows:
        return None
    if np.array_equal(out_combo, out_orig):
        return None

    inp = _clear_ext_side_keep_obstacles(out_combo, sep_col, bars_on_left, obstacle_color)

    if get_bg(inp) != 0:
        return None
    if obstacle_color not in colors_present(inp, exclude=[0]):
        return None

    left_colors = _side_colors(inp, 0, sep_col, 0)
    right_colors = _side_colors(inp, sep_col + 1, w, 0)
    if min(len(left_colors), len(right_colors)) != 1 or max(len(left_colors), len(right_colors)) < 2:
        return None
    if _bars_side_is_left(inp, sep_col, 0) != bars_on_left:
        return None
    if not np.array_equal(main(inp), out_combo):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
