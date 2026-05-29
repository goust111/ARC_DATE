# Variant of d8e07eb2
# Dimension: D3 — 参考表结构 (dst)
# Change: 参考表为4行4列的5x5单元格网格，每个单元格包含一个唯一颜色+形状的3x3字形，单元格之间用背景色(8)分隔... → 参考表含空单元格

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D3 variant
# ==================================
# a) 具体化与行为追踪
# - 画面结构仍是 30x22：上方 5 行查询带，中间两条色 6 分隔行之间是 4x4 的 5x5 表格，
#   下方 2 行是判定带。变化只发生在中间参考表：部分 5x5 单元格现在完全为空，整格都是背景色 8。
# - 查询带仍只放真实字形，所以它永远无法“引用”空单元格。输出仍会把匹配到的表格单元背景填绿。
# - 关键差异发生在“直线判定”这一步。原题里只要 4 个查询恰好落成同一整行/整列就成功；变体里若某行
#   有一个空格，那么该行最多只能匹配出 3 个绿色单元，哪怕这 3 个刚好排成直线，底部仍必须是红色。
# - 具体例子：某一表格行有 3 个字形 + 1 个空格。原流程在这里本来默认“完整行有 4 个可匹配单元”，
#   但现在新流程会看到这一行不是 open line；查询若正好复制这 3 个字形，只会高亮 3 格，判定失败。
#
# b) 复杂度校准
# - 空单元格数量限制在 2-4 个：肉眼容易看见参考表不再满格，又不会让表格过于稀疏。
# - 每个样例都要求同时存在：
#   1) 至少一条没有空格的整行/整列，保证 success 样例仍然可生成；
#   2) 至少一条恰好含 1 个空格的整行/整列，保证“被空位阻断”的失败样例可见。
# - 仍保持所有非空单元的 (颜色 + 3x3 形状) 唯一，避免匹配歧义。
#
# c) 原 main() 的隐式假设审查
# - 原代码用 `table_fg.min()` 推断表格左上角，前提是第一行和第一列都至少有一个非空单元。
#   D3 下这个前提不再稳固，所以这里改成按固定 4x4 / 5x5 几何与分隔行位置直接定位表格。
# - 原 success 条件只检查“4 个匹配格是否共行/共列”。这在旧题里足够，因为每条完整线都由 4 个字形组成。
#   D3 下需要显式区分 open line（4 格都非空）和 blocked line（含空格），因此增加 open row/open col 判断。
# - 原代码把空单元格当成普通背景略过，这一点在变体中仍然成立，因为空格本来就不能匹配。
#
# d) 放置合法性
# - 表格位置、单元尺寸、查询槽位都沿用原题，不发生重叠也不越界。
# - 空单元格只是不 paste 字形；非空格仍把 3x3 字形居中放进 5x5 单元。
# - 查询带只复制非空单元里的字形，因此 solver 不会遇到“查询引用空格”的非法状态。
#
# e) 视觉可观察性
# - 输入里直接能看到参考表有空洞；输出里可以看到三格成线却仍判红的 blocked-line 失败样例。
# - 与原题相比，解题者能从少量样例观察到：并不是“凑成直线就行”，而是“必须凑满没有空洞的整线”。
#
# f) 可视化整洁度
# - 仍只使用一种标记方式：匹配单元背景填绿，成功时再整块填绿查询带和判定带。
# - 空单元格保持纯背景，不与字形或高亮叠加，视觉上就是表中的缺口，层次清楚。
#
# g) 颜色变化可推断性
# - 变体没有引入新的颜色映射；绿色/红色语义与原题一致，只改变“何时成功”的结构条件。


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _cell_signature(
    g: np.ndarray,
    r0: int,
    c0: int,
    size: int,
    bg: int,
) -> tuple[tuple[int, int, int], ...] | None:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    mask = cell != int(bg)
    if not bool(mask.any()):
        return None
    return tuple(
        (int(r), int(c), int(cell[r, c]))
        for r, c in zip(*np.where(mask), strict=False)
    )


def _table_starts(top_sep: int, bottom_sep: int, width: int, cell_size: int, table_dim: int) -> tuple[list[int], list[int]]:
    row0 = int(bottom_sep) - int(table_dim) * int(cell_size)
    col0 = (int(width) - int(table_dim) * int(cell_size)) // 2
    row_starts = [row0 + i * cell_size for i in range(table_dim)]
    col_starts = [col0 + i * cell_size for i in range(table_dim)]
    return row_starts, col_starts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match query glyphs against a reference table that now includes blank cells.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - Above the first separator is a query strip of 5x5 glyph slots.
    - Between separators is a fixed 4x4 table of 5x5 cells; some cells contain a
      centered 3x3 colored glyph, while other cells are completely empty teal.

    Rule:
    - For every non-empty table cell whose glyph exactly matches one of the query
      glyphs, fill that cell's teal background with green.
    - A global success happens only when the matched cells occupy all four
      positions of one table row or one table column, so any line containing an
      empty table cell is impossible to complete.
    - On success, also green-fill the teal background in the query band and make
      the verdict band green; otherwise leave the query band unchanged and make
      the verdict band red.

    Key insight:
    Empty reference cells are structural blockers, not missing data. The task is
    still exact glyph lookup, but only fully populated lines can ever trigger the
    green verdict.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    bg = get_bg(g)
    cell_size = 5
    table_dim = 4

    sep_rows = _constant_non_bg_rows(g, bg)
    assert len(sep_rows) >= 2, "expected two full-width separator rows"
    top_sep = sep_rows[0]
    bottom_sep = sep_rows[-1]

    row_starts, col_starts = _table_starts(top_sep, bottom_sep, w, cell_size, table_dim)
    assert row_starts[0] > top_sep, "table rows must lie below the top separator"
    assert row_starts[-1] + cell_size <= bottom_sep, "table rows exceed table band"
    assert col_starts[0] >= 0 and col_starts[-1] + cell_size <= w, "table columns exceed grid width"

    query_sigs = [
        sig
        for c in col_starts
        if (sig := _cell_signature(g, 0, c, cell_size, bg)) is not None
    ]
    query_set = set(query_sigs)
    assert len(query_sigs) == len(query_set), "query glyphs should be distinct"

    table_sigs: dict[tuple[int, int], tuple[tuple[int, int, int], ...]] = {}
    occupied: set[tuple[int, int]] = set()
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c, cell_size, bg)
            if sig is None:
                continue
            occupied.add((int(ri), int(ci)))
            table_sigs[(int(ri), int(ci))] = sig
    assert occupied, "reference table must contain at least one glyph"

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for (ri, ci), sig in table_sigs.items():
        if sig not in query_set:
            continue
        selected.append((ri, ci))
        cell = out[row_starts[ri] : row_starts[ri] + cell_size, col_starts[ci] : col_starts[ci] + cell_size]
        cell[cell == bg] = 3

    open_rows = {
        ri
        for ri in range(table_dim)
        if all((ri, ci) in occupied for ci in range(table_dim))
    }
    open_cols = {
        ci
        for ci in range(table_dim)
        if all((ri, ci) in occupied for ri in range(table_dim))
    }

    selected_set = set(selected)
    success = False
    if len(query_sigs) == table_dim and len(selected) == table_dim:
        same_rows = {ri for ri, _ci in selected}
        same_cols = {ci for _ri, ci in selected}
        if len(same_rows) == 1:
            ri = next(iter(same_rows))
            success = ri in open_rows and all((ri, ci) in selected_set for ci in range(table_dim))
        elif len(same_cols) == 1:
            ci = next(iter(same_cols))
            success = ci in open_cols and all((ri, ci) in selected_set for ri in range(table_dim))

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = 3
        out[bottom_sep + 1 :, :] = 3
    else:
        out[bottom_sep + 1 :, :] = 2

    return out


# --------------- generator (o5x) ---------------

# -- Grid layout --
GRID_H, GRID_W = 30, 22
TOP_SEP_ROW = 5
BOTTOM_SEP_ROW = 27
QUERY_BAND_H = 5
VERDICT_ROW = 28

# -- Colors --
BG = 8
GREEN = 3
RED = 2
SEP = 6
GLYPH_COLORS = (0, 1, 2, 4, 5, 6, 7, 9)

# -- Table geometry --
TABLE_DIM = 4
CELL = 5
ROWS = (7, 12, 17, 22)
COLS = (1, 6, 11, 16)
ALL_CELLS = tuple((r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM))

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Empty cells / outcomes --
EMPTY_MIN, EMPTY_MAX = 2, 4
P_SUCCESS = 0.35
P_BLOCKED_FAIL = 0.6
P_QUERY_4_FAIL = 0.4

# -- Generation --
MAX_RETRIES = 400
EMPTY_LAYOUT_RETRIES = 300
FAIL_SAMPLE_RETRIES = 300


def _has_full_bbox(mask: int) -> bool:
    bits = [(mask >> i) & 1 for i in range(9)]
    return (
        any(bits[0:3])
        and any(bits[6:9])
        and any(bits[i] for i in (0, 3, 6))
        and any(bits[i] for i in (2, 5, 8))
    )


PATTERNS_BY_FILL = {
    k: tuple(m for m in range(1 << 9) if m.bit_count() == k and _has_full_bbox(m))
    for k in range(FILL_MIN, FILL_MAX + 1)
}


def _balanced_color_list(num_cells: int) -> list[int]:
    assert num_cells >= len(GLYPH_COLORS) * COLOR_MIN, "not enough cells for color minimums"
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = num_cells - len(GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _balanced_shape_masks(num_cells: int) -> list[int]:
    base_total = sum(SHAPE_BASE_COUNTS.values())
    assert num_cells >= base_total, "not enough cells for shape minimums"
    counts = dict(SHAPE_BASE_COUNTS)
    extras = num_cells - base_total
    for _ in range(extras):
        choices = [
            k
            for k in range(FILL_MIN, FILL_MAX + 1)
            if counts[k] < min(MAX_SHAPES_PER_FILL, len(PATTERNS_BY_FILL[k]))
        ]
        counts[random.choice(choices)] += 1

    masks: list[int] = []
    for k, n in counts.items():
        masks.extend(random.sample(PATTERNS_BY_FILL[k], n))
    random.shuffle(masks)
    return masks


def _glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _line_empty_counts(empty_cells: set[tuple[int, int]]) -> tuple[list[int], list[int]]:
    row_counts = [0] * TABLE_DIM
    col_counts = [0] * TABLE_DIM
    for r, c in empty_cells:
        row_counts[r] += 1
        col_counts[c] += 1
    return row_counts, col_counts


def _choose_empty_cells() -> set[tuple[int, int]]:
    for _ in range(EMPTY_LAYOUT_RETRIES):
        count = random.randint(EMPTY_MIN, EMPTY_MAX)
        empty_cells = set(random.sample(list(ALL_CELLS), count))
        row_counts, col_counts = _line_empty_counts(empty_cells)
        counts = row_counts + col_counts
        if max(counts) >= TABLE_DIM:
            continue
        if not any(v == 0 for v in counts):
            continue
        if not any(v == 1 for v in counts):
            continue
        return empty_cells
    raise RuntimeError("failed to place empty cells with both open and blocked lines")


def _open_lines(empty_cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    lines: list[list[tuple[int, int]]] = []
    for r in range(TABLE_DIM):
        line = [(r, c) for c in range(TABLE_DIM)]
        if all(cell not in empty_cells for cell in line):
            lines.append(line)
    for c in range(TABLE_DIM):
        line = [(r, c) for r in range(TABLE_DIM)]
        if all(cell not in empty_cells for cell in line):
            lines.append(line)
    return lines


def _blocked_lines(empty_cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    lines: list[list[tuple[int, int]]] = []
    for r in range(TABLE_DIM):
        line = [(r, c) for c in range(TABLE_DIM)]
        if sum(cell in empty_cells for cell in line) == 1:
            lines.append([cell for cell in line if cell not in empty_cells])
    for c in range(TABLE_DIM):
        line = [(r, c) for r in range(TABLE_DIM)]
        if sum(cell in empty_cells for cell in line) == 1:
            lines.append([cell for cell in line if cell not in empty_cells])
    return lines


def _is_success_cells(cells: list[tuple[int, int]], empty_cells: set[tuple[int, int]]) -> bool:
    if len(cells) != TABLE_DIM:
        return False
    cell_set = set(cells)
    rows = {r for r, _c in cell_set}
    cols = {c for _r, c in cell_set}
    if len(rows) == 1:
        r = next(iter(rows))
        return all((r, c) not in empty_cells for c in range(TABLE_DIM)) and cell_set == {
            (r, c) for c in range(TABLE_DIM)
        }
    if len(cols) == 1:
        c = next(iter(cols))
        return all((r, c) not in empty_cells for r in range(TABLE_DIM)) and cell_set == {
            (r, c) for r in range(TABLE_DIM)
        }
    return False


def _generic_fail_cells(nonempty_cells: list[tuple[int, int]], empty_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    count = 4 if random.random() < P_QUERY_4_FAIL else 3
    for _ in range(FAIL_SAMPLE_RETRIES):
        cells = random.sample(nonempty_cells, count)
        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if _is_success_cells(cells, empty_cells):
            continue
        if len(rows) == 1 or len(cols) == 1:
            continue
        random.shuffle(cells)
        return cells
    raise RuntimeError("failed to sample a generic fail query set")


def _expected_output(grid: np.ndarray, cells: list[tuple[int, int]], success: bool) -> np.ndarray:
    out = grid.copy()
    for r_i, c_i in cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell[cell == BG] = GREEN
    if success:
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == BG] = GREEN
        out[VERDICT_ROW:, :] = GREEN
    else:
        out[VERDICT_ROW:, :] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    empty_cells = _choose_empty_cells()
    nonempty_cells = [cell for cell in ALL_CELLS if cell not in empty_cells]
    open_lines = _open_lines(empty_cells)
    blocked_lines = _blocked_lines(empty_cells)
    assert open_lines and blocked_lines

    masks = _balanced_shape_masks(len(nonempty_cells))
    colors = _balanced_color_list(len(nonempty_cells))
    glyphs: dict[tuple[int, int], np.ndarray] = {}
    for (r_i, c_i), mask, color in zip(nonempty_cells, masks, colors, strict=True):
        glyph = _glyph(mask, color)
        glyphs[(r_i, c_i)] = glyph
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    if random.random() < P_SUCCESS:
        success = True
        query_cells = random.choice(open_lines).copy()
    else:
        success = False
        if random.random() < P_BLOCKED_FAIL:
            query_cells = random.choice(blocked_lines).copy()
        else:
            query_cells = _generic_fail_cells(nonempty_cells, empty_cells)

    random.shuffle(query_cells)
    for slot, cell in enumerate(query_cells):
        paste(grid, glyphs[cell], 0, COLS[slot])

    expected = _expected_output(grid, query_cells, success)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 D3 input")
