# Combo variant of 1ae2feb7 (index 32)
# combo_spec: D1/0+D14/1+D15/2
# Name: 对角线十字涂色冲突留白

from __future__ import annotations

import math

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color

# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
# - 保留原题的整体骨架：一列近乎完整的竖直分隔列把网格分成条形侧和扩展侧；输入里条形侧有 2-4 个
#   活跃行，每行 1-2 个水平单色条，扩展侧为空。
# - 融合后的单次“周期落点”不再是同一行上的单像素，而是沿 45 度方向前进的中心点，再把该中心点扩成
#   只允许落在扩展侧内的 5 格十字：中心、上、下、左、右。靠近分隔列或边界时，超出扩展侧的手臂被裁掉。
# - 冲突按“claim 计数”一次性结算，而不是最近条优先。任何格子只要被两个及以上十字标记覆盖，就留作背景。
# - 具体小例子：若条形在左侧、分隔列在 c=5，某行 r 上有长度 2 的近条和长度 3 的远条。
#   原题会在 (r,6),(r,8),(r,10)... 与 (r,6),(r,9),(r,12)... 上按近条优先落单点。
#   本融合版先把两条射线的中心点改成对角线：近条中心是 (r+1,6),(r+3,8),(r+5,10)...，
#   远条中心是 (r+1,6),(r+4,9),(r+7,12)...。每个中心再扩成十字，所以第一次共同中心
#   (r+1,6) 会让 (r,6),(r+1,6),(r+2,6),(r+1,7) 等扩展侧格子都成为多重 claim，最终一起留白；
#   后续不重叠的位置才保留对应颜色。差异不再只是“同行点列”，而是跨行斜向十字和规则性的空洞。
#
# b) 复杂度校准
# - 方向只保留两种全局情况：左侧条对应右下斜线，右侧条对应左下斜线，没有按颜色或按行切换方向。
# - 十字形固定为半径 1 的 5 格，不引入多种模板，避免规则分叉太多。
# - 生成器把活跃行上限收紧到 4，并偏向较短条长，让对角线重复与十字冲突都能在有限宽度内出现，但不至于
#   让扩展侧完全糊成一片。
#
# c) 隐式假设审查
# - `_find_sep_col` 仍依赖“分隔列是近乎全满的同色竖列”，融合版不改这点。
# - `bars_on_left` 的质量判断仍依赖输入扩展侧为空；生成器继续 output-first 后清空扩展侧保证成立。
# - `_iter_runs_1d` 仍假设条形是单行水平连续段，这也不变。
# - 原 solver 中“逐条写入且已上色就跳过”的隐式前提在这里完全失效：现在既有跨行斜向扩散，也有十字
#   扩张，还要求多重 claim 留白，所以必须先全局收集所有 claim，再统一按次数结算。
# - 融合后新增的关键假设是：十字只在扩展侧结算，不能回写到分隔列或条形侧，否则会破坏可读性并掩盖输入
#   结构。solver 通过 `_in_ext_side` 显式裁剪，generator 也按同样规则模拟。
#
# d) 放置合法性
# - 活跃行只从 `0..h-2` 里选，保证每条对角线至少有首个中心点 `(row+1, first_ext_col)` 落在网格内。
# - 条形仍按靠分隔列向外堆叠，长度和总长受 bar-side 容量约束，不重叠、不越界。
# - 十字 claim 统一经过边界和扩展侧裁剪，因此不会写到分隔列、条形侧或网格外。
# - generator 仍采用 output-first：先放条、再按 solver 同步的 claim 规则生成输出、最后清空扩展侧得到
#   输入，并用 `main(input) == output` 做最终验证。
#
# e) 视觉可观察性
# - 2-4 组样例里，人能直接看到三个稳定信号：射线沿斜线下行、每个周期不是单点而是小十字、重叠处形成
#   系统性的空白洞。规则不依赖颜色编号或隐藏排序。
# - 为防退化，生成器强制样本同时具备：至少一个因十字扩张而新增的上色格、至少一个额外冲突留白格、
#   至少一个落在非源条行上的扩展侧着色格。
#
# f) 可视化整洁度
# - 十字会让局部密度上升，所以保持原题的“活跃行之间至少隔一行”，并把条长偏向小值，让图案呈现清楚的
#   几条斜向轨迹，而不是整片密集填满。
# - 十字不允许跨越分隔列，避免把输入线索区和输出区搅在一起。
#
# g) 变化的可推断性
# - 颜色不重映射；唯一上色格继承首次 claim 的条形颜色，多重 claim 一律留白。
# - 颜色推断仍由“哪个条的节奏命中该位置”决定，新增难点只来自二维空间中的十字交叠，而不是额外的颜色表。


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


def _in_ext_side(col: int, sep_col: int, bars_on_left: bool, w: int) -> bool:
    if bars_on_left:
        return sep_col < col < w
    return 0 <= col < sep_col


def _collect_bars(
    g: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> list[tuple[int, int, int, int, int, int]]:
    bars: list[tuple[int, int, int, int, int, int]] = []
    h, w = g.shape
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


def _iter_diagonal_centers(
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


def _iter_cross_cells(
    row: int,
    col: int,
    sep_col: int,
    bars_on_left: bool,
    h: int,
    w: int,
):
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = row + dr
        cc = col + dc
        if 0 <= rr < h and _in_ext_side(cc, sep_col, bars_on_left, w):
            yield rr, cc


def _collect_claims(
    bars: list[tuple[int, int, int, int, int, int]],
    sep_col: int,
    bars_on_left: bool,
    h: int,
    w: int,
    *,
    cross: bool,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    counts: dict[tuple[int, int], int] = {}
    first_color: dict[tuple[int, int], int] = {}
    for _dist, row, _c0, _c1, length, color in bars:
        for rr, cc in _iter_diagonal_centers(row, sep_col, bars_on_left, length, h, w):
            cells = (
                _iter_cross_cells(rr, cc, sep_col, bars_on_left, h, w)
                if cross
                else ((rr, cc),)
            )
            for cell in cells:
                counts[cell] = counts.get(cell, 0) + 1
                first_color.setdefault(cell, color)
    return counts, first_color


def _apply_claims(
    out: np.ndarray,
    counts: dict[tuple[int, int], int],
    first_color: dict[tuple[int, int], int],
    bg: int,
) -> None:
    for (rr, cc), count in counts.items():
        if count == 1 and int(out[rr, cc]) == bg:
            out[rr, cc] = first_color[(rr, cc)]


# Overview:
#   Detect the separator and the bar side, then build all diagonal cross-shaped claims globally.
# Input structure:
#   One near-solid vertical separator column; all bars are horizontal runs on exactly one side;
#   the extension side is blank in valid inputs.
# Rule:
#   Each bar of length L emits diagonal centers at 45 degrees with period L. Every center expands to
#   a 5-cell cross clipped to the extension side only. A cell is colored iff exactly one cross claims it;
#   any multi-claim cell stays background.
# Key insight:
#   The old nearest-first overwrite logic no longer applies. Because diagonal motion and cross expansion
#   create cross-row interactions, the correct solver must count simultaneous claims before painting.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    bars = _collect_bars(g, sep_col, bars_on_left, bg)
    counts, first_color = _collect_claims(
        bars,
        sep_col,
        bars_on_left,
        h,
        w,
        cross=True,
    )
    _apply_claims(out, counts, first_color, bg)
    return out


H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 2, 4

P_TWO_BARS = 0.6
P_SEP_GAP = 0.15
P_SWAP_COLORS = 0.5
P_SEP_COLUMN_GAP = 0.2

MIN_FORCED_CONFLICT_ROWS = 1
MAX_ATTEMPTS = 500
_GEN_INPUT_CALL_COUNT = 0


def _choose_active_rows(rng: np.random.Generator, usable_rows: int, count: int) -> list[int]:
    """Pick `count` row indices in [0, usable_rows) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(usable_rows - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _pick_second_color(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
) -> int:
    others = [c for c in palette if c != primary]
    return int(rng.choice(others))


def _conflict_length_pairs(capacity: int, ext_side_len: int) -> list[tuple[int, int]]:
    max_len = min(capacity, 5)
    pairs = [
        (a, b)
        for a in range(1, max_len + 1)
        for b in range(1, max_len + 1)
        if a + b <= capacity and a != b
    ]
    rich_pairs = [pair for pair in pairs if math.lcm(pair[0], pair[1]) <= ext_side_len]
    return rich_pairs if rich_pairs else pairs


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    ext_side_len: int,
    *,
    force_conflict: bool = False,
) -> tuple[list[tuple[int, int]], int]:
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        raise ValueError("bar-side capacity must stay positive")

    want_two_bars = force_conflict or (capacity >= 2 and rng.random() < P_TWO_BARS)
    if not want_two_bars or capacity < 2:
        max_single = min(capacity, 5)
        length = int(rng.integers(1, max_single + 1))
        return [(length, primary)], sep_gap

    pair_choices = _conflict_length_pairs(capacity, ext_side_len)
    if not pair_choices:
        max_single = min(capacity, 5)
        length = int(rng.integers(1, max_single + 1))
        return [(length, primary)], sep_gap

    near_len, far_len = pair_choices[int(rng.integers(0, len(pair_choices)))]
    if rng.random() < 0.5:
        near_len, far_len = far_len, near_len

    colors = [primary, _pick_second_color(rng, palette, primary)]
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
) -> list[tuple[int, int, int, int, int, int]]:
    records: list[tuple[int, int, int, int, int, int]] = []
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
            out[row, cursor : cursor + length] = color
            dist = c0 - (sep_col + 1)
            records.append((dist, row, c0, c1, length, color))
            cursor = c1 + 1
    return records


def _apply_variant_extension(
    out: np.ndarray,
    bars: list[tuple[int, int, int, int, int, int]],
    sep_col: int,
    bars_on_left: bool,
    bg: int,
    *,
    cross: bool,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    h, w = out.shape
    counts, first_color = _collect_claims(
        bars,
        sep_col,
        bars_on_left,
        h,
        w,
        cross=cross,
    )
    _apply_claims(out, counts, first_color, bg)
    return counts, first_color


def _variant_metrics(
    inp: np.ndarray,
    out: np.ndarray,
    bars: list[tuple[int, int, int, int, int, int]],
    sep_col: int,
    bars_on_left: bool,
    active_rows: list[int],
) -> tuple[int, int, int]:
    h, w = out.shape
    center_counts, center_colors = _collect_claims(
        bars,
        sep_col,
        bars_on_left,
        h,
        w,
        cross=False,
    )
    baseline = inp.copy()
    _apply_claims(baseline, center_counts, center_colors, bg=0)

    ext_mask = np.zeros_like(out, dtype=bool)
    if bars_on_left:
        ext_mask[:, sep_col + 1 :] = True
    else:
        ext_mask[:, :sep_col] = True

    arm_only_paint = int(np.count_nonzero((out != 0) & (baseline == 0) & ext_mask))
    cross_counts, _ = _collect_claims(
        bars,
        sep_col,
        bars_on_left,
        h,
        w,
        cross=True,
    )
    visible_conflict_cells = 0
    for (rr, cc), count in cross_counts.items():
        if count <= 1 or int(out[rr, cc]) != 0:
            continue
        painted_neighbors = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and ext_mask[nr, nc] and int(out[nr, nc]) != 0:
                painted_neighbors += 1
        if painted_neighbors >= 2:
            visible_conflict_cells += 1

    active_mask = np.zeros(h, dtype=bool)
    active_mask[active_rows] = True
    inactive_row_paint = int(np.count_nonzero((out != 0) & ext_mask & (~active_mask[:, None])))
    return arm_only_paint, visible_conflict_cells, inactive_row_paint


def _gen_attempt(forced_bars_on_left: bool | None = None) -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))
    ext_side_len = w - 1 - bar_side_len

    if forced_bars_on_left is None:
        bars_on_left = bool(rng.integers(0, 2) == 0)
    else:
        bars_on_left = forced_bars_on_left
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    usable_rows = h - 1
    max_active = min(ACTIVE_MAX, (usable_rows + 1) // 2)
    if max_active < ACTIVE_MIN:
        return None
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, usable_rows, active_count)
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_candidates = [r for r in inactive_rows if r == 0 or r == h - 1]
        if not gap_candidates:
            gap_candidates = inactive_rows
        gap_row = int(gap_candidates[int(rng.integers(0, len(gap_candidates)))])
        out[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    forced_conflict_rows = set(active_rows[: min(MIN_FORCED_CONFLICT_ROWS, len(active_rows))])
    if len(active_rows) > 1 and rng.random() < 0.5:
        forced_conflict_rows.add(active_rows[int(rng.integers(1, len(active_rows)))])

    bar_records: list[tuple[int, int, int, int, int, int]] = []
    conflict_row_count = 0
    for idx, row in enumerate(active_rows):
        bars, sep_gap = _sample_row_bars(
            rng,
            palette,
            row_primary[idx],
            bar_side_len,
            ext_side_len,
            force_conflict=row in forced_conflict_rows,
        )
        if len(bars) >= 2:
            conflict_row_count += 1
        bar_records.extend(_place_bars(out, row, sep_col, bars_on_left, bars, sep_gap))

    if conflict_row_count < MIN_FORCED_CONFLICT_ROWS:
        return None

    bar_records.sort(key=lambda t: (t[0], t[1], t[2]))
    _apply_variant_extension(out, bar_records, sep_col, bars_on_left, bg=0, cross=True)

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

    arm_only_paint, visible_conflict_cells, inactive_row_paint = _variant_metrics(
        inp,
        out,
        bar_records,
        sep_col,
        bars_on_left,
        active_rows,
    )
    if arm_only_paint <= 0 or visible_conflict_cells <= 0 or inactive_row_paint <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    global _GEN_INPUT_CALL_COUNT
    forced_bars_on_left = (_GEN_INPUT_CALL_COUNT % 2) == 0
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(forced_bars_on_left=forced_bars_on_left)
        if grid is not None:
            _GEN_INPUT_CALL_COUNT += 1
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
