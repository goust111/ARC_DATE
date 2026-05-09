# Combo variant of 1ae2feb7 (index 34)
# combo_spec: D5/1+D7/0+D10/0
# Name: 多色条远优先处理后色条变色
#
# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 保留原题框架：一列竖直 separator，把网格分成 bar_side 和 ext_side；输入中 ext_side 全是背景色。
#    - D10 要求每个活跃行不再只有 1-2 条 bar，而是固定 3 或 4 条相邻单色 bar，按“靠近 separator
#      -> 远离 separator”的顺序摆放，且同一行颜色两两不同。
#    - D7 把 ext_side 的处理顺序从“近优先”翻转为“远优先”。因此同一行中最远的 bar 先在 ext_side
#      的偏移 0, L, 2L, ... 上落点；靠近 separator 的 bar 只能补空位，不能覆盖已上色像素。
#    - D5 再追加一次统一 recolor：ext_side 全部涂完后，bar_side 上所有原始 bar 像素一起改成
#      separator 的颜色。
#    - 具体单行例子（bars_on_left）：
#        输入:   [7 7][4 4 4][9 9 9 9] | 2 | 0 0 0 0 0 0 0 0
#               近->远长度 = [2,3,4]，颜色 = [7,4,9]
#        原题:   先 7 占 offset 0,2,4,6；再 4 补 3；再 9 只剩 5,7 ...
#        融合:   先 9 占 offset 0,4；再 4 补 3,6；再 7 补 2。
#               最后 bar_side 的 [7 7][4 4 4][9 9 9 9] 全部重涂成 2。
#        输出:   [2 2][2 2 2][2 2 2 2] | 2 | 9 0 7 4 9 0 4 0
#    - 差异来自两个层面同时可见：
#      1) ext_side 上第一格和若干重叠格由“最远 bar”赢得，完全反转原题优先级；
#      2) 输出里 bar_side 不再保留原始 bar 颜色，而是与 separator 同色。
# b) 复杂度校准
#    - 每个活跃行固定 3-4 条 bar，避免退化回原题的 1-2 条情形。
#    - 同一行长度互异、颜色互异，并要求每条 bar 在 far-first 输出里至少留下 1 个可见 ext_side 像素，
#      否则“有很多 bar 但只有 1 个颜色真正可见”会变成假复杂度。
#    - 4 条 bar 只在容量允许时出现；否则回退到 3 条，控制自由度，减少无效重试。
# c) 隐式假设审查
#    - `_find_sep_col` 依赖“几乎整列同色”的 separator；generator 继续保证 separator 整列填色，
#      或仅在非活跃的顶/底行留一个缺口。
#    - `bars_on_left` 通过 separator 两侧的非背景像素总量判断 bar 所在侧；输入里 ext_side 仍保持全 0，
#      因此前提成立。
#    - `_iter_runs_1d` 将每个同色连续段视为一条 bar。融合后 recolor 会让输出里的 bar 融合到 separator，
#      所以 solver 必须先从输入收集 bar 坐标，再做 recolor；不能从输出反推。
#    - 原题全局按 distance 升序排序编码了“近优先”；融合版必须改为 distance 降序，且 no-overwrite
#      规则保留不变，这正是顺序翻转能被观察到的原因。
#    - 多个改动叠加后新增的关键前提是：同一行相邻 bar 在输入中不能同色，否则 run 会被合并。generator
#      通过“同一行颜色互异”显式保证这一点。
# d) 放置合法性
#    - bar_side 宽度提高到至少 6，以容纳 3 条互异正长度 bar；4 条 bar 只在容量至少 10 时采样。
#    - 同一行 bar 仍然彼此相邻，只允许整体与 separator 有 0 或 1 格空隙，因此不会互相重叠或越界。
#    - 活跃行之间继续至少隔 1 空行，避免不同行 ext_side 图样互相干扰。
# e) 视觉可观察性
#    - 远优先不依赖颜色编号，而只依赖“距离 separator 的远近”，这是可视几何关系。
#    - 因为每个活跃行至少有 3 个异色 bar，且所有 bar 都会争夺 ext_side 的第一格，所以 far-first 与
#      near-first 在每个活跃行都会出现可见差异，尤其是第一格颜色直接从“最近 bar”变成“最远 bar”。
#    - recolor 目标固定为 separator 的颜色，结构线索明确，不需要推断任意颜色映射。
# f) 可视化整洁度
#    - 叠加只发生在单行的 ext_side，且依旧通过“不覆盖已有颜色”裁决，因此图样虽然更密，但不会变成实心块。
#    - bar_side 的 recolor 统一、稳定，视觉上形成“输入是多彩条，输出变成 separator 色短条”的清晰对比。
# g) 变化的可推断性
#    - 输出中的原 bar 颜色只会出现在 ext_side，这恰好提示“源条颜色被投射到另一侧，而源条本身变色”。
#    - 由于所有 bar 最终都变成 separator 色，解题者需要同时对齐 input 的多彩 bar 与 output 的 ext_side
#      颜色点列，这个映射具有明显的结构支撑而非任意查表。
# 9) main() 的新隐式假设审查
#    - `main()` 假设每个活跃行的所有 bar 都在 separator 同一侧，且 ext_side 输入全背景。generator 保证。
#    - `main()` 假设每条 bar 长度为正且 row-local 顺序可由距离唯一确定。generator 只采样正长度且长度互异。
#    - `main()` 假设 recolor 目标就是 separator 颜色。generator 始终使用单一 separator 色，并在输出验证
#      所有 bar 像素都被改写为该颜色。

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations

import numpy as np

from arc_utils import colors_present, get_bg


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


def _collect_row_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> dict[int, list[tuple[int, int, int, int, int]]]:
    """Return row-local bars as (dist, c0, c1, length, color), ordered near->far."""
    h, w = g.shape
    row_bars: dict[int, list[tuple[int, int, int, int, int]]] = {}
    for r in range(h):
        runs = (
            _iter_runs_1d(g[r, :], 0, sep_col, bg=bg)
            if bars_on_left
            else _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg)
        )
        for c0, c1, col in runs:
            length = c1 - c0 + 1
            dist = (sep_col - 1) - c1 if bars_on_left else c0 - (sep_col + 1)
            row_bars.setdefault(r, []).append((dist, c0, c1, length, col))
    for r in row_bars:
        row_bars[r].sort(key=lambda t: t[0])
    return row_bars


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int, int, int, int]],
    *,
    far_first: bool,
    bg: int,
) -> None:
    w = out.shape[1]
    ordered = list(reversed(bars)) if far_first else list(bars)
    for _dist, _c0, _c1, length, col in ordered:
        if length <= 0:
            continue
        positions = range(sep_col + 1, w, length) if bars_on_left else range(sep_col - 1, -1, -length)
        for cc in positions:
            if int(out[row, cc]) == bg:
                out[row, cc] = col


def _recolor_bars(
    out: np.ndarray,
    row_bars: dict[int, list[tuple[int, int, int, int, int]]],
    sep_color: int,
) -> None:
    for r, bars in row_bars.items():
        for _dist, c0, c1, _length, _col in bars:
            out[r, c0 : c1 + 1] = sep_color


def _solve(grid: np.ndarray, *, far_first: bool, recolor: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h == 0 or w == 0:
        return g.copy()

    bg = get_bg(g)
    sep_col, sep_color = _find_sep_col(g, bg)
    if sep_col is None or sep_color is None:
        return g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    row_bars = _collect_row_bars(g, sep_col, bars_on_left, bg)
    out = g.copy()
    for r, bars in row_bars.items():
        _paint_extension(out, r, sep_col, bars_on_left, bars, far_first=far_first, bg=bg)

    if recolor:
        _recolor_bars(out, row_bars, sep_color)
    return out


# Overview:
#   Solve the combo variant by treating each active row as a stack of 3-4 adjacent bars that
#   project periodic dots onto the opposite side, with the farthest bar claiming overlaps first,
#   then recoloring all source bars to the separator color.
# Input structure:
#   A mostly solid vertical separator column splits the grid into a bar side and an empty
#   extension side. Each active row contains 3 or 4 adjacent monochrome bars on exactly one side
#   of the separator; inactive rows are background except for the separator.
# Rule:
#   For every active row, collect its bars from the input, order them by distance from the
#   separator from far to near, and let each bar paint its own period on the extension side using
#   its length as the step while never overwriting non-background pixels. After all extension
#   painting is done, recolor every original bar cell to the separator color.
# Key insight:
#   The extension pattern is determined jointly by bar length and bar distance to the separator:
#   all bars aim at the first extension cell, so reversing the order makes the farthest bar win
#   those conflicts; the original bar colors survive only in the extension because the bars
#   themselves are recolored afterward.
def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, far_first=True, recolor=True)


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 6, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 4, 14

PALETTE_MIN, PALETTE_MAX = 6, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 5

P_FOUR_BARS = 0.4
P_SEP_GAP = 0.12
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 400


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _visible_bar_count(ext_side_len: int, lengths_near_to_far: tuple[int, ...], *, far_first: bool) -> int:
    filled: set[int] = set()
    visible = 0
    ordered = tuple(reversed(lengths_near_to_far)) if far_first else lengths_near_to_far
    for length in ordered:
        contributed = False
        for offset in range(0, ext_side_len, length):
            if offset not in filled:
                filled.add(offset)
                contributed = True
        if contributed:
            visible += 1
    return visible


@lru_cache(maxsize=None)
def _feasible_length_orders(
    capacity: int,
    ext_side_len: int,
    bar_count: int,
) -> tuple[tuple[int, ...], ...]:
    orders: list[tuple[int, ...]] = []
    for comb in combinations(range(1, capacity + 1), bar_count):
        if sum(comb) > capacity:
            continue
        for perm in permutations(comb):
            if _visible_bar_count(ext_side_len, tuple(map(int, perm)), far_first=True) != bar_count:
                continue
            orders.append(tuple(map(int, perm)))
    return tuple(orders)


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    bar_side_len: int,
    ext_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap_order = [1, 0] if rng.random() < P_SEP_GAP else [0, 1]
    for sep_gap in sep_gap_order:
        capacity = bar_side_len - sep_gap
        if capacity < 6:
            continue

        choices: list[tuple[int, tuple[int, ...]]] = []
        orders3 = _feasible_length_orders(capacity, ext_side_len, 3)
        if orders3:
            choices.append((3, orders3[int(rng.integers(0, len(orders3)))]))
        orders4 = _feasible_length_orders(capacity, ext_side_len, 4)
        if orders4:
            choices.append((4, orders4[int(rng.integers(0, len(orders4)))]))
        if not choices:
            continue

        choice4 = next((item for item in choices if item[0] == 4), None)
        choice3 = next((item for item in choices if item[0] == 3), None)
        if choice4 is not None and rng.random() < P_FOUR_BARS:
            bar_count, lengths = choice4
        elif choice3 is not None:
            bar_count, lengths = choice3
        else:
            bar_count, lengths = choice4  # type: ignore[misc]

        colors = list(map(int, rng.choice(palette, size=bar_count, replace=False)))
        return list(zip(lengths, colors)), sep_gap

    raise ValueError("no feasible multi-bar row configuration")


def _place_bars(
    grid: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
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


def _ext_slice(sep_col: int, bars_on_left: bool, w: int) -> slice:
    return slice(sep_col + 1, w) if bars_on_left else slice(0, sep_col)


def _separator_gap_candidates(h: int, active_rows: set[int]) -> list[int]:
    return [r for r in (0, h - 1) if r not in active_rows]


def _all_row_colors_visible(
    out: np.ndarray,
    row_specs: dict[int, tuple[list[tuple[int, int]], int]],
    sep_col: int,
    bars_on_left: bool,
) -> bool:
    ext = _ext_slice(sep_col, bars_on_left, out.shape[1])
    for row, (bars, _sep_gap) in row_specs.items():
        visible = set(map(int, out[row, ext].tolist()))
        visible.discard(0)
        needed = {color for _length, color in bars}
        if not needed.issubset(visible):
            return False
    return True


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

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_rows_set = set(active_rows)

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color

    gap_candidates = _separator_gap_candidates(h, active_rows_set)
    if gap_candidates and rng.random() < P_SEP_COLUMN_GAP:
        inp[int(rng.choice(gap_candidates)), sep_col] = 0

    row_specs: dict[int, tuple[list[tuple[int, int]], int]] = {}
    for row in active_rows:
        try:
            bars, sep_gap = _sample_row_bars(rng, palette, bar_side_len, ext_side_len)
        except ValueError:
            return None
        row_specs[row] = (bars, sep_gap)
        _place_bars(inp, row, sep_col, bars_on_left, bars, sep_gap)

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None

    solved = main(inp)
    alt = _solve(inp, far_first=False, recolor=True)

    if np.array_equal(solved, alt):
        return None
    if not _all_row_colors_visible(solved, row_specs, sep_col, bars_on_left):
        return None

    bar_mask = (inp != 0) & (inp != sep_color)
    if not np.any(bar_mask):
        return None
    if not np.all(solved[bar_mask] == sep_color):
        return None

    ext = _ext_slice(sep_col, bars_on_left, w)
    if np.count_nonzero(solved[:, ext] != alt[:, ext]) == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
