# Variant of 1ae2feb7
# Dimension: D10 — 色条数量与排列 (src)
# Change: 每行1-2个单色色条，相邻排列（无间隔），紧邻分隔条；2-5个活跃行，行间至少隔1空行... → 每行3个或更多色条
#
# Design analysis
# source_task: none
# concrete_form:
#   - 仍然只有一条竖直分隔列，输入的 ext_side 仍然全为背景色。
#   - 活跃行仍然彼此至少隔 1 空行，但每个活跃行现在固定放 3 或 4 个相邻色条。
#   - 同一行内每个色条的颜色都不同、长度也都不同；最靠近分隔条的色条先向 ext_side 发射周期射线。
#   - 输出中同一行的 ext_side 会出现 3-4 组不同步长的周期点列，后处理的远端色条不能覆盖前面已上色的位置。
# behavior_trace:
#   - 例：分隔条在中间，左侧某活跃行从近到远的长度为 [4, 3, 2, 1]，颜色为 [a, b, c, d]。
#   - 原题 generator 最多只会给 [4, 3] 或单条 [4]；main 只需处理 1-2 条射线。
#   - 本变体中 main 必须在同一行追踪 4 条射线：
#       a 在 ext 偏移 0,4,8...
#       b 试图画 0,3,6...，但 0 被 a 占据，因此只在 3,6... 中的空位留下
#       c 试图画 0,2,4,6...，只能补上 2、可能再补 10...
#       d 试图画 0,1,2,3...，只能在先前都没碰到的位置落点
#   - 差异不在“周期规则”本身，而在同一行可同时参与竞争的色条数显著增加，ext 纹理更密、更交错。
# complexity_calibration:
#   - 为了让多射线交互可见，生成器强制每个活跃行都有 3 或 4 条色条，禁止退化回 1-2 条。
#   - 同一行内要求每条色条都能在 ext_side 留下至少 1 个可见像素，避免“有 4 条但实际只看见 1 条颜色”的假复杂度。
#   - 4 条色条只在容量允许时出现，避免把自由度推得过高导致大量重试。
# implicit_assumption_audit:
#   - `_find_sep_col`: 依赖“分隔列几乎整列同色”。变体不改这一点，generator 继续保证整列或仅缺 1 格。
#   - `bars_on_left` 判定: 依赖输入里只有 bars side 含非背景色块。ext_side 仍在输入中清空，所以前提成立。
#   - 原版把所有 run 放进一个全局列表再按距离排序；这在多行时是无害的，但对 D10 而言“逐行追踪多条射线”更清晰。
#     新版改成先按行收集、再在行内按距离排序，显式体现每行 3-4 条 run 的处理顺序。
#   - `step = length`: 依赖长度为正。generator 只采样正长度，且同一行长度互异。
# placement_legality:
#   - bars side 宽度提升到至少 6，以便容纳 3 条不同长度；若要 4 条，则需要至少 10 格。
#   - 同一行的色条仍然彼此相邻，只允许整体与分隔条有 0 或 1 格间隙，因此不会重叠、不会越界。
#   - active rows 继续满足间隔 >= 2（即中间至少 1 空行），避免行间视觉干扰。
# visual_observability:
#   - 规则完全由可见几何决定：色条离分隔条的先后顺序、每条色条长度、以及 ext_side 上的周期点。
#   - 没有使用颜色 id 排序，也没有任何隐藏映射；人类能直接从“近的先画，远的补空位”归纳规则。
# cleanliness:
#   - 叠加只发生在同一活跃行的 ext_side，且仍通过“不覆盖已有颜色”裁决，画面不会变成无法区分的实心块。
# color_inference:
#   - 颜色只复制，不做变换；颜色数量增加只是为了区分更多色条，不需要额外推断颜色映射。
# main_new_assumptions_review:
#   - `main()` 现在假设 bars side 的每个非背景 run 都是独立色条；generator 保证同色不会在同一行相邻出现从而误合并。
#   - `main()` 现在按“每行局部排序”处理 run；generator 保证每行所有色条都在同一侧且互不重叠，因此不会丢失顺序信息。

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
        if row[c] == bg:
            c += 1
            continue
        col = int(row[c])
        c0 = c
        while c < c_hi and row[c] == col:
            c += 1
        yield c0, c - 1, col


def _collect_row_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> dict[int, list[tuple[int, int, int]]]:
    row_bars: dict[int, list[tuple[int, int, int]]] = {}
    h, w = g.shape
    for r in range(h):
        runs = (
            _iter_runs_1d(g[r, :], 0, sep_col, bg=bg)
            if bars_on_left
            else _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg)
        )
        for c0, c1, col in runs:
            length = c1 - c0 + 1
            dist = (sep_col - 1) - c1 if bars_on_left else c0 - (sep_col + 1)
            row_bars.setdefault(r, []).append((dist, length, col))
    for r in row_bars:
        row_bars[r].sort(key=lambda t: t[0])
    return row_bars


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h == 0 or w == 0:
        return g.copy()

    bg = get_bg(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    row_bars = _collect_row_bars(g, sep_col, bars_on_left, bg)
    for r, bars in row_bars.items():
        for _dist, length, col in bars:
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

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

# Raised lower bound so every active row can host at least 3 distinct lengths.
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


def _visible_color_count(ext_side_len: int, lengths: tuple[int, ...]) -> int:
    filled = set()
    visible = 0
    for length in lengths:
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
            if perm[0] == 1:
                continue
            if _visible_color_count(ext_side_len, perm) != bar_count:
                continue
            orders.append(tuple(map(int, perm)))
    return tuple(orders)


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    bar_side_len: int,
    ext_side_len: int,
) -> tuple[list[tuple[int, int]], int]:
    best_choice: tuple[list[tuple[int, int]], int] | None = None
    sep_gap_order = [1, 0] if rng.random() < P_SEP_GAP else [0, 1]

    for sep_gap in sep_gap_order:
        capacity = bar_side_len - sep_gap
        if capacity < 6:
            continue

        choices: list[tuple[int, tuple[int, ...]]] = []
        three_orders = _feasible_length_orders(capacity, ext_side_len, 3)
        if three_orders:
            choices.append((3, tuple(three_orders[int(rng.integers(0, len(three_orders)))])))
        four_orders = _feasible_length_orders(capacity, ext_side_len, 4)
        if four_orders:
            choices.append((4, tuple(four_orders[int(rng.integers(0, len(four_orders)))])))
        if not choices:
            continue

        four_choice = next(((count, vals) for count, vals in choices if count == 4), None)
        three_choice = next(((count, vals) for count, vals in choices if count == 3), None)
        if four_choice is not None and rng.random() < P_FOUR_BARS:
            bar_count, lengths = four_choice
        elif three_choice is not None:
            bar_count, lengths = three_choice
        else:
            bar_count, lengths = four_choice  # type: ignore[misc]

        colors = list(map(int, rng.choice(palette, size=bar_count, replace=False)))
        best_choice = (list(zip(lengths, colors)), sep_gap)
        break

    if best_choice is None:
        raise ValueError("no feasible multi-bar row configuration")
    return best_choice


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


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
) -> None:
    w = out.shape[1]
    for length, color in bars:
        positions = range(sep_col + 1, w, length) if bars_on_left else range(sep_col - 1, -1, -length)
        for cc in positions:
            if out[row, cc] == 0:
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
    ext_side_len = w - 1 - bar_side_len

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_rows_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_rows_set]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for row in active_rows:
        try:
            bars, sep_gap = _sample_row_bars(rng, palette, bar_side_len, ext_side_len)
        except ValueError:
            return None
        row_bars_map[row] = bars
        _place_bars(out, row, sep_col, bars_on_left, bars, sep_gap)

    for row, bars in row_bars_map.items():
        _paint_extension(out, row, sep_col, bars_on_left, bars)

    inp = out.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

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
