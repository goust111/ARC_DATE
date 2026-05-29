# Combo variant of 1ae2feb7 (index 18)
# combo_spec: D1/0+D9/2+D14/1
# Name: 缺口分隔条下对角线十字涂色

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the combo variant
# a) 具体化与行为追踪
# - 原题：某行条形长度为 L，若分隔列在其右侧，则从 `(r, sep+1)` 开始每隔 L 个格子在同一行落 1 个点。
# - 本融合：只有 `grid[r, sep]` 真有分隔色的条形行才允许发射；每次发射的第一个落点改为
#   `(r+1, sep+1)`（右扩时）或 `(r+1, sep-1)`（左扩时），之后每次同时沿行和列各走 L 步，
#   形成 45 度对角线锚点；每个锚点再把 `(rr-1, cc),(rr, cc),(rr+1, cc)` 三格都尝试上色。
# - 例：bars_on_left, `sep=5`, 某发射行 `r=2` 上近侧条长 `L=3`。原题会涂 `(2,6),(2,9),(2,12)`；
#   现在会在锚点 `(3,6),(6,9),(9,12)` 处各画一根竖向 3 格短线。若 `r=4` 这一行在分隔列上是缺口，
#   则其条形完全不发射，但仍可能被 `r=2` 的十字扩散染到。
# b) 复杂度校准
# - 三个变化必须同时可见：生成器强制存在发射行和阻断行，且至少有一个样本中阻断行若取消缺口会带来
#   额外涂色；同时要求融合输出和“直线十字版”“对角线单点版”都不同，避免退化成单维变体。
# - 条形仍限制为每行 1-2 段、全局最多 8 段，颜色唯一，既保证分隔列可辨识，也避免画面失控。
# c) 隐式假设审查
# - 原 `_find_sep_col` 依赖“分隔列几乎填满整列”，这里失效，改成：寻找某种颜色仅出现在单列、且其某一侧
#   在输入中整片为空，从而同时识别分隔列与条形所在侧。
# - 原 solver 默认“所有条形行都发射”，这里失效，必须只收集 `sep_rows` 上的条形。
# - 原 solver 默认“每个周期只写一个格”，这里失效，必须逐格检查十字三格并保持不覆盖。
# - 原 generator 可以逐行独立模拟；这里不行，因为对角线十字会跨行挡色，必须像 solver 一样先全局排序
#   再统一落色。
# d) 放置合法性
# - 条形仍只放在分隔列一侧；输入扩展侧始终清空。
# - 活跃行只从 `0..h-2` 里选，保证每个发射条至少存在第一个对角线锚点。
# - 十字涂色逐格做越界和背景检查，因此不会覆盖已有条形或分隔像素，也不会越界。
# e) 视觉可观察性
# - 输入里分隔列显式断裂，输出里只有对应发射行延伸；落点离开原行并排成斜线，且每个落点是短竖线，
#   因此三种变化都能被肉眼看到。
# f) 可视化整洁度
# - 活跃行保持至少隔 1 个空行，条形端点仍贴近分隔列；输出虽然跨行，但只使用固定斜率和固定 3 格标记，
#   没有多种方向混杂。
# g) 变化可推断性
# - 不做颜色映射。每段条形保留自身颜色，颜色线索只服务于“谁发射、谁先占位、谁被阻断”，人类可从
#   结构而非颜色编号本身归纳规则。
#
# 新 `main()` 的额外假设与 generator 保证
# - 假设分隔色是输入中唯一一个“跨多行但只占一列”的颜色；generator 让每个条段颜色全局唯一来保证。
# - 假设输入扩展侧全空；generator 在 output-first 之后整侧清空。
# - 假设至少 2 个发射行，否则稀疏分隔列不稳定；generator 强制 `EMIT_ROWS_MIN = 2`。

BarRecord = tuple[int, int, int, int, int, int]  # dist, row, c0, c1, length, color


def _find_sep_info(g: np.ndarray, bg: int) -> tuple[int | None, int | None, bool | None]:
    """Find the sparse separator by looking for a repeated single-column color plus a blank ext side."""
    h, w = g.shape
    for color in colors_present(g, exclude=[bg]):
        coords = np.argwhere(g == int(color))
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) < 2 or len(cols) != 1:
            continue
        sep_col = int(cols[0])
        if sep_col + 1 < w and np.all(g[:, sep_col + 1 :] == bg):
            return sep_col, int(color), True
        if sep_col > 0 and np.all(g[:, :sep_col] == bg):
            return sep_col, int(color), False
    return None, None, None


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


def _iter_diagonal_positions(
    row: int,
    sep_col: int,
    bars_on_left: bool,
    step: int,
    h: int,
    w: int,
):
    if step <= 0:
        return
    rr = row + 1
    cc = sep_col + 1 if bars_on_left else sep_col - 1
    dc = 1 if bars_on_left else -1
    while 0 <= rr < h and 0 <= cc < w:
        yield rr, cc
        rr += step
        cc += dc * step


def _paint_vertical_mark(out: np.ndarray, row: int, col: int, color: int, bg: int) -> None:
    h, _w = out.shape
    for rr in (row - 1, row, row + 1):
        if 0 <= rr < h and int(out[rr, col]) == bg:
            out[rr, col] = color


def _collect_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
    rows: list[int] | np.ndarray,
) -> list[BarRecord]:
    bars: list[BarRecord] = []
    _h, w = g.shape
    for r in map(int, rows):
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


def _apply_extension(
    out: np.ndarray,
    bars: list[BarRecord],
    sep_col: int,
    bars_on_left: bool,
    bg: int,
    *,
    diagonal: bool,
    cross: bool,
) -> None:
    h, w = out.shape
    for _dist, r, _c0, _c1, length, color in bars:
        if length <= 0:
            continue
        if diagonal:
            positions = _iter_diagonal_positions(r, sep_col, bars_on_left, length, h, w)
            for rr, cc in positions:
                if cross:
                    _paint_vertical_mark(out, rr, cc, color, bg)
                elif int(out[rr, cc]) == bg:
                    out[rr, cc] = color
        else:
            if bars_on_left:
                cols = range(sep_col + 1, w, length)
            else:
                cols = range(sep_col - 1, -1, -length)
            for cc in cols:
                if cross:
                    _paint_vertical_mark(out, r, cc, color, bg)
                elif int(out[r, cc]) == bg:
                    out[r, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend only separator-aligned bars as diagonal rays whose landing points expand into vertical crosses.

    Input structure:
    - A background grid split by a sparse vertical separator color that appears only on some active rows.
    - All bar segments lie on one side of that separator; the opposite side is fully blank in the input.
    - Rows whose separator cell is missing may still contain bars, but they are blocked rows.

    Rule:
    - Detect the broken separator column and determine which side contains the bars from the blank extension side.
    - Collect bars only from rows where the separator color is present.
    - Process bars nearest-to-separator first; each bar of length L emits diagonal landing points starting one step
      beyond the separator and then every L diagonal steps toward the extension side.
    - At every landing point, paint a 3-cell vertical mark centered on that point, never overwriting a non-background cell.

    Key insight:
    - The separator holes are not decorative: they gate which rows may emit at all, while the emitted pattern itself
      is no longer row-local because diagonal stepping and cross-shaped marks spread color into neighboring rows.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    sep_col, sep_color, bars_on_left = _find_sep_info(g, bg)
    if sep_col is None or sep_color is None or bars_on_left is None:
        return g.copy()

    out = g.copy()
    sep_rows = np.flatnonzero(g[:, sep_col] == sep_color)
    bars = _collect_bars(g, sep_col, bars_on_left, bg, sep_rows)
    _apply_extension(out, bars, sep_col, bars_on_left, bg, diagonal=True, cross=True)
    return out


H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 4, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 5, 14

BAR_ROWS_MIN, BAR_ROWS_MAX = 4, 5
EMIT_ROWS_MIN = 2

P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5

MAX_ATTEMPTS = 600


def _choose_active_rows(rng: np.random.Generator, usable_rows: int, count: int) -> list[int]:
    """Pick `count` row indices in [0, usable_rows) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(usable_rows - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_colors(rng: np.random.Generator, seg_count: int) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    pool = [c for c in range(1, 10) if c != sep_color]
    seg_colors = list(map(int, rng.choice(pool, size=seg_count, replace=False)))
    return sep_color, seg_colors


def _sample_row_shapes(
    rng: np.random.Generator,
    active_count: int,
    bar_side_len: int,
) -> list[tuple[list[int], int]]:
    """Return per-row bar lengths (near->far) and separator gap, with total segments <= 8."""
    extra_budget = min(active_count, 8 - active_count)
    two_count = int(rng.integers(0, extra_budget + 1))
    two_rows = set(map(int, rng.choice(active_count, size=two_count, replace=False))) if two_count else set()

    shapes: list[tuple[list[int], int]] = []
    for idx in range(active_count):
        sep_gap = 1 if rng.random() < P_SEP_GAP else 0
        capacity = bar_side_len - sep_gap
        if capacity <= 0:
            return []

        if idx not in two_rows or capacity < 2:
            length = int(rng.integers(1, min(capacity, 5) + 1))
            shapes.append(([length], sep_gap))
            continue

        near_len = int(rng.integers(1, min(capacity - 1, 4) + 1))
        far_cap = min(capacity - near_len, 5)
        if far_cap <= 0:
            shapes.append(([near_len], sep_gap))
            continue
        far_len = int(rng.integers(1, far_cap + 1))
        lengths = [near_len, far_len]
        if rng.random() < P_SWAP_COLORS:
            lengths = lengths[::-1]
        shapes.append((lengths, sep_gap))
    return shapes


def _place_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> list[BarRecord]:
    records: list[BarRecord] = []
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, color in bars:
            c1 = cursor - 1
            c0 = cursor - length
            out[row, c0:cursor] = color
            dist = (sep_col - 1) - c1
            records.append((dist, row, c0, c1, length, color))
            cursor = c0
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            c0 = cursor
            c1 = cursor + length - 1
            out[row, c0 : c1 + 1] = color
            dist = c0 - (sep_col + 1)
            records.append((dist, row, c0, c1, length, color))
            cursor = c1 + 1
    return records


def _clear_ext_side(g: np.ndarray, sep_col: int, bars_on_left: bool) -> np.ndarray:
    out = g.copy()
    if bars_on_left:
        out[:, sep_col + 1 :] = 0
    else:
        out[:, :sep_col] = 0
    return out


def _anchor_count(record: BarRecord, sep_col: int, bars_on_left: bool, shape: tuple[int, int]) -> int:
    _dist, row, _c0, _c1, length, _color = record
    h, w = shape
    return sum(1 for _ in _iter_diagonal_positions(row, sep_col, bars_on_left, length, h, w))


def _fusion_metrics(
    inp: np.ndarray,
    fused_out: np.ndarray,
    all_bars: list[BarRecord],
    emit_rows: set[int],
    sep_col: int,
    bars_on_left: bool,
) -> tuple[int, int, int]:
    emit_bars = [rec for rec in all_bars if rec[1] in emit_rows]

    straight_cross = inp.copy()
    _apply_extension(straight_cross, emit_bars, sep_col, bars_on_left, 0, diagonal=False, cross=True)

    diagonal_single = inp.copy()
    _apply_extension(diagonal_single, emit_bars, sep_col, bars_on_left, 0, diagonal=True, cross=False)

    ungated = inp.copy()
    _apply_extension(ungated, all_bars, sep_col, bars_on_left, 0, diagonal=True, cross=True)

    blocked_effect = int(np.count_nonzero(ungated != fused_out))
    diagonal_effect = int(np.count_nonzero(straight_cross != fused_out))
    cross_effect = int(np.count_nonzero(diagonal_single != fused_out))
    return blocked_effect, diagonal_effect, cross_effect


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

    usable_rows = h - 1
    max_rows = min(BAR_ROWS_MAX, usable_rows // 2 + usable_rows % 2)
    if max_rows < BAR_ROWS_MIN:
        return None
    active_count = int(rng.integers(BAR_ROWS_MIN, max_rows + 1))
    active_rows = _choose_active_rows(rng, usable_rows, active_count)

    emit_count = int(rng.integers(EMIT_ROWS_MIN, active_count))
    emit_rows = set(map(int, rng.choice(active_rows, size=emit_count, replace=False)))
    blocked_rows = [r for r in active_rows if r not in emit_rows]
    if not blocked_rows:
        return None

    row_shapes = _sample_row_shapes(rng, active_count, bar_side_len)
    if not row_shapes or len(row_shapes) != active_count:
        return None

    seg_count = sum(len(lengths) for lengths, _ in row_shapes)
    if seg_count > 8:
        return None
    sep_color, seg_colors = _sample_colors(rng, seg_count)

    out = np.zeros((h, w), dtype=int)
    all_bar_records: list[BarRecord] = []
    color_cursor = 0
    for row, (lengths, sep_gap) in zip(active_rows, row_shapes):
        colors = seg_colors[color_cursor : color_cursor + len(lengths)]
        color_cursor += len(lengths)
        bars = list(zip(lengths, colors))
        all_bar_records.extend(_place_bars(out, int(row), sep_col, bars_on_left, bars, sep_gap))
        if row in emit_rows:
            out[int(row), sep_col] = sep_color

    all_bar_records.sort(key=lambda t: (t[0], t[1], t[2]))
    emit_bar_records = [rec for rec in all_bar_records if rec[1] in emit_rows]
    _apply_extension(out, emit_bar_records, sep_col, bars_on_left, 0, diagonal=True, cross=True)

    inp = _clear_ext_side(out, sep_col, bars_on_left)

    if get_bg(inp) != 0:
        return None
    if int(np.count_nonzero(inp[:, sep_col] == sep_color)) < EMIT_ROWS_MIN:
        return None
    if not all(int(inp[r, sep_col]) == 0 for r in blocked_rows):
        return None
    if not np.array_equal(main(inp), out):
        return None

    repeat_bars = sum(
        _anchor_count(rec, sep_col, bars_on_left, out.shape) >= 2
        for rec in emit_bar_records
    )
    if repeat_bars <= 0:
        return None

    blocked_effect, diagonal_effect, cross_effect = _fusion_metrics(
        inp, out, all_bar_records, emit_rows, sep_col, bars_on_left
    )
    if blocked_effect <= 0 or diagonal_effect <= 0 or cross_effect <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
