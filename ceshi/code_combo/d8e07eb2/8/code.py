# Combo variant of d8e07eb2 (index 8)
# combo_spec: D3/1+D5/3
# Name: 含空格参考表+多数规则判定

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for combo D3/1 + D5/3
# ======================================
# a) 具体化与行为追踪
# - 画面仍是 30x22 固定布局：上方 5 行查询带，中间两条色 6 分隔行之间是 4x4 的 5x5 单元格表，
#   下方 2 行是判定带。D3/1 让参考表里出现空单元格；D5/3 把判定从“整行/整列全满”改成“某行/列多数匹配”。
# - 这里把“多数”具体化为：对任一 4 格行/列，匹配数 > 未匹配数；因为每条线固定长 4，所以阈值就是 3。
#   关键融合决策是：空单元格仍然占据该线的一个位置，但它永远不可能被匹配，因此空格按未匹配计数。
# - 例子 1：某一行有 3 个真实字形 + 1 个空格，查询正好复制这 3 个字形。原题与 D3 单变体都失败，
#   因为不满足完整 4 格；融合后这一行是 3 匹配对 1 未匹配，所以成功，查询带与判定带都填绿。
# - 例子 2：某一列有 2 个字形 + 2 个空格，查询复制这 2 个字形再加一个别处字形。该列只得到 2:2 平手，
#   不超过未匹配数，因此仍失败。这样样例里能直接看见“空格不是忽略项，而是反对票”。
#
# b) 复杂度校准
# - 空单元格数量控制在 3-4 个，且要求 8 条候选线（4 行 4 列）里同时出现 0 空格、1 空格、2 空格三种档位。
#   这样同一任务族里会稳定出现 open line、blocked-majority line、blocked-tie line。
# - 成功样例主要来自两类：3/4 的开放行列，或 3 个实字形 + 1 空格的阻塞行列。失败样例则包含 2/4 平手线
#   和散点线，避免规则退化成“只看查询数量”或“只看是否共线”。
# - 非空单元仍保持 (颜色 + 3x3 形状) 全局唯一，避免匹配歧义。
#
# c) 隐式假设审查
# - 原 solver 用 `table_fg.min()` 推断表格左上角，默认第一行第一列一定有字形；空单元格会破坏这个前提。
#   融合版改为根据分隔行和固定 4x4 几何直接定位。
# - 原 solver 把 success 绑到“查询数=4 且同一整行/整列”；D5/3 下这个假设失效，因为 3 个查询也可能成功。
# - 原 solver 实际只检查 selected 是否同一行/列，没有显式把空位纳入计数。融合版改为逐条 4 格线计票，
#   未被匹配的位置统一视为未匹配，其中自然包含空格与未被查询的真实字形。
# - 新规则还隐含一个前提：查询字形彼此不同，且参考表中每个非空签名唯一。生成器继续强制这两点。
#
# d) 放置合法性
# - 表格位置仍固定在 rows=(7,12,17,22), cols=(1,6,11,16)，每个单元 5x5，不重叠不越界。
# - 空单元格通过“不 paste 任何字形”实现；查询带只复制非空单元格中的 glyph，因此不会引用空位。
# - 查询最多 4 个，按左到右槽位放入上方 4 个 5x5 槽；当查询数为 3 时，最后一个槽保持背景。
#
# e) 视觉可观察性
# - 成功样例会频繁出现“只有 3 个查询，但输出判绿”的情况，解题者能明显看出已不再要求 4 个全满。
# - 失败样例会出现“某条线上的所有真实字形都被选中，但因为同线还有 2 个空格，所以仍判红”的情况，
#   从而逼迫观察者意识到空格参与多数比较。
# - 通过混合 open-majority 与 blocked-majority success，样例能区分“缺 1 个真实字形”和“缺的是空位”。
#
# f) 可视化整洁度
# - 输出标记方式不增加新图层：仍只把匹配到的表格单元背景填绿，成功时再整块填绿查询带与判定带。
# - 空单元格保持纯背景色，不会出现对象叠加；视觉上的变化集中在表格洞位与底部红/绿判定，结构清楚。
#
# g) 变化的可推断性
# - 颜色语义没有新增映射：绿色始终表示命中的表格单元与成功 verdict，红色表示失败 verdict。
# - 真正变化的是结构计数规则，而且阈值只有一个明确档位：4 格线中至少 3 格命中才成功。
# - 由于生成器会稳定产出 3 命中成功、2 命中失败、以及含空格的多数/平手对照，少量样例就能归纳规则。
#
# h) 对新 main() 假设的反查
# - `main()` 现在假设表格始终是固定 4x4、单元大小固定 5、查询槽与表格列对齐；生成器固定遵守该几何。
# - `main()` 假设每个查询 glyph 至多匹配一个表格单元；生成器保证非空单元签名唯一。
# - `main()` 的 success 只依赖“某条 4 格线上的匹配数是否 >= 3”；生成器显式构造 blocked-majority、
#   open-majority、blocked-tie、scatter 四类模式，确保所有分支都被覆盖。


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


def _table_starts(bottom_sep: int, width: int, cell_size: int, table_dim: int) -> tuple[list[int], list[int]]:
    row0 = int(bottom_sep) - int(table_dim) * int(cell_size)
    col0 = (int(width) - int(table_dim) * int(cell_size)) // 2
    row_starts = [row0 + i * cell_size for i in range(table_dim)]
    col_starts = [col0 + i * cell_size for i in range(table_dim)]
    return row_starts, col_starts


def _line_match_max(cells: set[tuple[int, int]], table_dim: int) -> int:
    row_best = max(sum((r, c) in cells for c in range(table_dim)) for r in range(table_dim))
    col_best = max(sum((r, c) in cells for r in range(table_dim)) for c in range(table_dim))
    return max(row_best, col_best)


def _is_majority_success(cells: set[tuple[int, int]], table_dim: int) -> bool:
    best = _line_match_max(cells, table_dim)
    return best > table_dim - best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match query glyphs into a partly blank reference table and judge by line majority.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - Above the first separator is a query strip of four 5x5 slots containing
      three or four copied glyphs.
    - Between separators is a fixed 4x4 table of 5x5 cells; some cells contain a
      centered 3x3 colored glyph and some cells are entirely blank teal.

    Rule:
    - For every non-empty table cell whose glyph exactly appears in the query
      strip, fill that cell's teal background with green.
    - Then inspect each table row and column as a 4-position line. If any line
      has more matched positions than unmatched positions, the puzzle succeeds.
      Blank table cells still count as unmatched positions because they occupy a
      line slot but can never be matched.
    - On success, also green-fill the teal background in the query band and make
      the verdict band green; otherwise leave the query band unchanged and make
      the verdict band red.

    Key insight:
    The answer is no longer about completing a whole line. You must count votes
    on each 4-cell row or column, and blank reference cells contribute negative
    pressure because they reduce how many matches that line can possibly earn.
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

    row_starts, col_starts = _table_starts(bottom_sep, w, cell_size, table_dim)
    assert row_starts[0] > top_sep, "table rows must lie below the top separator"
    assert row_starts[-1] + cell_size <= bottom_sep, "table rows exceed table band"
    assert col_starts[0] >= 0 and col_starts[-1] + cell_size <= w, "table columns exceed grid width"

    query_sigs = [
        sig
        for c in col_starts
        if (sig := _cell_signature(g, 0, c, cell_size, bg)) is not None
    ]
    query_set = set(query_sigs)
    assert query_sigs, "query strip must contain at least one glyph"
    assert len(query_sigs) == len(query_set), "query glyphs should be distinct"

    table_sigs: dict[tuple[int, int], tuple[tuple[int, int, int], ...]] = {}
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c, cell_size, bg)
            if sig is not None:
                table_sigs[(int(ri), int(ci))] = sig
    assert table_sigs, "reference table must contain at least one glyph"
    assert len(table_sigs) == len(set(table_sigs.values())), "table glyphs should be unique"

    out = g.copy()
    selected: set[tuple[int, int]] = set()
    for (ri, ci), sig in table_sigs.items():
        if sig not in query_set:
            continue
        selected.add((ri, ci))
        cell = out[row_starts[ri] : row_starts[ri] + cell_size, col_starts[ci] : col_starts[ci] + cell_size]
        cell[cell == bg] = 3

    success = _is_majority_success(selected, table_dim)
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
EMPTY_MIN, EMPTY_MAX = 3, 4
P_SUCCESS = 0.45
P_BLOCKED_SUCCESS = 0.65
P_BLOCKED_FAIL = 0.5
P_QUERY_4_FAIL = 0.35

# -- Generation --
MAX_RETRIES = 500
EMPTY_LAYOUT_RETRIES = 400
SAMPLE_RETRIES = 400


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


def _line_profiles(empty_cells: set[tuple[int, int]]) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []
    for r in range(TABLE_DIM):
        line = [(r, c) for c in range(TABLE_DIM)]
        occupied = [cell for cell in line if cell not in empty_cells]
        profiles.append({"axis": "row", "index": r, "cells": line, "occupied": occupied})
    for c in range(TABLE_DIM):
        line = [(r, c) for r in range(TABLE_DIM)]
        occupied = [cell for cell in line if cell not in empty_cells]
        profiles.append({"axis": "col", "index": c, "cells": line, "occupied": occupied})
    return profiles


def _choose_empty_cells() -> set[tuple[int, int]]:
    for _ in range(EMPTY_LAYOUT_RETRIES):
        count = random.randint(EMPTY_MIN, EMPTY_MAX)
        empty_cells = set(random.sample(list(ALL_CELLS), count))
        profiles = _line_profiles(empty_cells)
        empty_counts = [TABLE_DIM - len(profile["occupied"]) for profile in profiles]
        if max(empty_counts) > 2:
            continue
        if not any(v == 0 for v in empty_counts):
            continue
        if not any(v == 1 for v in empty_counts):
            continue
        if not any(v == 2 for v in empty_counts):
            continue
        return empty_cells
    raise RuntimeError("failed to place empty cells with open, blocked-majority, and blocked-tie lines")


def _sample_extras(
    base_cells: list[tuple[int, int]],
    nonempty_cells: list[tuple[int, int]],
    total_count: int,
) -> list[tuple[int, int]] | None:
    base_set = set(base_cells)
    pool = [cell for cell in nonempty_cells if cell not in base_set]
    if total_count < len(base_cells) or len(pool) < total_count - len(base_cells):
        return None
    for _ in range(SAMPLE_RETRIES):
        extras = random.sample(pool, total_count - len(base_cells))
        cells = list(base_cells) + extras
        if len(set(cells)) != total_count:
            continue
        if _is_majority_success(set(cells), TABLE_DIM):
            continue
        random.shuffle(cells)
        return cells
    return None


def _choose_success_cells(profiles: list[dict[str, object]]) -> list[tuple[int, int]]:
    blocked = [list(profile["occupied"]) for profile in profiles if len(profile["occupied"]) == 3]
    open_lines = [list(profile["occupied"]) for profile in profiles if len(profile["occupied"]) == 4]
    if blocked and (not open_lines or random.random() < P_BLOCKED_SUCCESS):
        cells = random.choice(blocked).copy()
    else:
        line = random.choice(open_lines)
        cells = random.sample(line, 3)
    random.shuffle(cells)
    assert _is_majority_success(set(cells), TABLE_DIM)
    return cells


def _choose_fail_cells(
    profiles: list[dict[str, object]],
    nonempty_cells: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    tie_lines = [list(profile["occupied"]) for profile in profiles if len(profile["occupied"]) == 2]
    open_lines = [list(profile["occupied"]) for profile in profiles if len(profile["occupied"]) == 4]

    modes = ["scatter"]
    if tie_lines:
        modes.append("blocked_tie")
    if open_lines:
        modes.append("open_tie")

    for _ in range(SAMPLE_RETRIES):
        mode = random.choice(modes if random.random() < 0.8 else ["scatter"])
        total_count = 4 if random.random() < P_QUERY_4_FAIL else 3

        if mode == "blocked_tie":
            cells = _sample_extras(random.choice(tie_lines), nonempty_cells, total_count)
            if cells is not None:
                return cells
        elif mode == "open_tie":
            base = random.sample(random.choice(open_lines), 2)
            cells = _sample_extras(base, nonempty_cells, total_count)
            if cells is not None:
                return cells
        else:
            cells = random.sample(nonempty_cells, total_count)
            if _is_majority_success(set(cells), TABLE_DIM):
                continue
            random.shuffle(cells)
            return cells

    raise RuntimeError("failed to sample a fail query set")


def _expected_output(grid: np.ndarray, cells: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for r_i, c_i in cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell[cell == BG] = GREEN
    if _is_majority_success(set(cells), TABLE_DIM):
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
    profiles = _line_profiles(empty_cells)

    masks = _balanced_shape_masks(len(nonempty_cells))
    colors = _balanced_color_list(len(nonempty_cells))
    glyphs: dict[tuple[int, int], np.ndarray] = {}
    for (r_i, c_i), mask, color in zip(nonempty_cells, masks, colors, strict=True):
        glyph = _glyph(mask, color)
        glyphs[(r_i, c_i)] = glyph
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    if random.random() < P_SUCCESS:
        query_cells = _choose_success_cells(profiles)
    else:
        query_cells = _choose_fail_cells(profiles, nonempty_cells)

    for slot, cell in enumerate(query_cells):
        paste(grid, glyphs[cell], 0, COLS[slot])

    expected = _expected_output(grid, query_cells)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
