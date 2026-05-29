# Combo variant of d8e07eb2 (index 4)
# combo_spec: D1/1+D3/1
# Name: 形状匹配+参考表含空格

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) 具体化与行为追踪
# - 画面骨架保持原题：30x22 画布，上方 5 行是查询带，第 5 行与第 27 行是整行分隔色 6，
#   中间是 4x4 的 5x5 参考表，下方 2 行是判定带。融合后的变化都发生在“参考表怎么读”。
# - D1/1 让匹配键从“颜色+形状”改成“只看 3x3 二值形状”，所以表中现在必须故意放入
#   同形异色的字形家族；否则忽略颜色不会造成任何可见差异。
# - D3/1 让参考表出现全背景空格。空格不能被查询命中，而且任何含空格的行/列都不可能构成成功线。
# - 一个具体追踪例子：
#   1) 某 blocked row 里有三格非空，形状家族 A 占两格、家族 B 占一格，第四格为空。
#   2) 查询带放入 [A, A, B] 或 [A, A, B, B]，其中 A/B 都只复制各自一个 exemplar。
#   3) 原题逻辑会按颜色+形状做精确查找，若 A 的隐藏同形异色副本与查询颜色不同，就不会被选中；
#      融合后它会被一起高亮，所以输出比原题多出“跨颜色额外命中”。
#   4) 即使三格高亮刚好排成一行，融合后仍失败，因为该行还有一个空格；这体现 D3/1 的阻断作用。
# - success 样例则选择一条没有空格的 open line，并把它分成两个 2 格形状家族。查询重复这两个家族的
#   exemplar 后，solver 会仅凭形状高亮整条 4 格线，并把查询带与判定带一起染绿。
#
# b) 复杂度校准
# - 空格数量限制为 2-4 个：能稳定看出“表里有洞”，又不会让 4x4 表格过稀。
# - 被查询的形状家族只做 2 格重复，避免同一形状扩散到太多位置，否则 shape-only 匹配会造成大面积高亮，
#   反而淹没“整行/整列”和“空格阻断”的主逻辑。
# - fail 主要使用 blocked line，而不是任意散点失败。这样解题者更容易从少量样例看出：
#   红色不仅因为没排成线，也可能因为线里有缺口。
#
# c) 隐式假设审查
# - 原 solver 通过 `table_fg.min()` 推断表格起点，默认左上角附近一定有字形；空格打破这个前提，
#   所以这里改成用固定几何（4x4、5x5、居中布局）直接求行列起点。
# - 原 `_cell_signature()` 把颜色写进签名；D1/1 下必须改成只记录前景坐标，否则同形异色无法合并。
# - 原 `assert len(query_sigs) == len(query_set)` 假定查询字形互不重复；融合后会故意重复同一家族的 exemplar，
#   所以该断言必须移除。
# - 原 success 条件 `len(selected) == len(query_sigs)` 默认“一条查询最多命中一格表项”；shape-only 匹配会让
#   一个查询家族命中多个表格位置，因此新条件必须显式检查：被选中的格子集合是否恰好等于某条全非空 4 格线。
# - 多改动叠加后的新冲突是：shape-only 会扩散命中范围，empty cells 会减少可完成线。generator 必须保证
#   被查询家族的全部副本要么正好构成目标 open line，要么正好落在被空格截断的 blocked line 内。
#
# d) 放置合法性
# - 所有表格单元仍严格放在 `ROWS x COLS` 的 5x5 槽位，不重叠、不越界。
# - 空格单元就是完全不 paste 字形；非空单元里的字形仍是 5x5 里居中的 3x3。
# - 查询带只复制非空表格单元对应的 exemplar，且仅放在前 3 或 4 个槽位，尾部空槽保持背景色。
#
# e) 视觉可观察性
# - 输入里能直接看见参考表缺格；输出里会出现“一个查询颜色没有直接出现在某格，但那一格仍被高亮”的现象，
#   从而暴露“只看形状、不看颜色”。
# - fail 样例优先选用 blocked line，使人类能在 2-4 组样例中观察到：三格排成直线仍然可能判红，因为整线并不完整。
#
# f) 可视化整洁度
# - 仍只使用一种局部标记：把匹配单元的背景涂绿。绿色不会覆盖字形本身，只填充原本的背景像素。
# - 空格保持纯背景，不叠加字形，也不被错误地高亮，表中的“洞”与“命中格”层次清晰。
#
# g) 变化的可推断性
# - 本融合没有新的颜色映射规则；颜色只作为干扰项，真正有效的是 3x3 轮廓是否相同。
# - 为了让这一点可归纳，generator 保证每个样例至少有一个被查询家族在表中存在同形异色副本，且该副本会被输出额外高亮。
#
# h) generator 对 main() 新假设的保障
# - main() 假设参考表是固定 4x4 几何；generator 始终使用原题的固定布局常量。
# - main() 假设至少存在一个非空参考字形；generator 的空格上限是 4，因此至少保留 12 个非空单元。
# - main() 假设 success 时被选中的 4 格就是某条全非空整线；generator 通过目标线专用家族布局保证这一点。
# - main() 假设 fail 时即使查询槽位填满，匹配结果也不会等于某条可完成整线；generator 只用 blocked line 作为
#   fail 目标，因此被选中格数最多为 3，天然不会误判为成功。


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _cell_shape_signature(
    g: np.ndarray,
    r0: int,
    c0: int,
    size: int,
    bg: int,
) -> tuple[tuple[int, int], ...] | None:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    mask = cell != int(bg)
    if not bool(mask.any()):
        return None
    return tuple((int(r), int(c)) for r, c in zip(*np.where(mask), strict=False))


def _table_starts(width: int, cell_size: int, table_dim: int, bottom_sep: int) -> tuple[list[int], list[int]]:
    row0 = int(bottom_sep) - int(table_dim) * int(cell_size)
    col0 = (int(width) - int(table_dim) * int(cell_size)) // 2
    row_starts = [row0 + i * cell_size for i in range(table_dim)]
    col_starts = [col0 + i * cell_size for i in range(table_dim)]
    return row_starts, col_starts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match query glyph silhouettes against a partly empty reference table.

    Input structure:
    - A teal canvas is split by two full-width separator rows.
    - The top band holds up to four 5x5 query slots containing centered 3x3 glyphs.
    - The middle band is a fixed 4x4 table of 5x5 cells; some cells contain a
      colored glyph, while some cells are entirely empty teal.

    Rule:
    - Ignore glyph colors and compare only the 3x3 foreground shape.
    - Green-fill every non-empty table cell whose shape appears in the query band.
    - If the matched cells are exactly all four positions of one completely
      non-empty table row or one completely non-empty table column, also
      green-fill the query-band background and the verdict band; otherwise only
      the verdict band turns red.

    Key insight: Query glyphs are just shape exemplars. Same-shaped cells of
    other colors still count, but empty table cells are blockers that keep any
    incomplete row or column from ever becoming a success.
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

    row_starts, col_starts = _table_starts(w, cell_size, table_dim, bottom_sep)
    assert row_starts[0] > top_sep, "table must lie below the top separator"
    assert row_starts[-1] + cell_size <= bottom_sep, "table extends past bottom separator"
    assert col_starts[0] >= 0 and col_starts[-1] + cell_size <= w, "table extends past grid width"

    query_shapes = [
        sig
        for c in col_starts
        if (sig := _cell_shape_signature(g, 0, c, cell_size, bg)) is not None
    ]
    query_shape_set = set(query_shapes)

    occupied: set[tuple[int, int]] = set()
    table_shapes: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_shape_signature(g, r, c, cell_size, bg)
            if sig is None:
                continue
            occupied.add((int(ri), int(ci)))
            table_shapes[(int(ri), int(ci))] = sig
    assert occupied, "reference table must contain at least one glyph"

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for (ri, ci), sig in table_shapes.items():
        if sig not in query_shape_set:
            continue
        selected.append((ri, ci))
        cell = out[row_starts[ri] : row_starts[ri] + cell_size, col_starts[ci] : col_starts[ci] + cell_size]
        cell[cell == bg] = 3

    selected_set = set(selected)
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

    success = False
    if len(query_shapes) == table_dim and len(selected_set) == table_dim:
        rows = {ri for ri, _ci in selected_set}
        cols = {ci for _ri, ci in selected_set}
        if len(rows) == 1:
            ri = next(iter(rows))
            success = ri in open_rows and selected_set == {(ri, ci) for ci in range(table_dim)}
        elif len(cols) == 1:
            ci = next(iter(cols))
            success = ci in open_cols and selected_set == {(ri, ci) for ri in range(table_dim)}

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = 3
        out[bottom_sep + 1 :, :] = 3
    else:
        out[bottom_sep + 1 :, :] = 2

    return out


# --------------- generator ---------------

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
MAX_SHAPES_PER_FILL = 5

# -- Color distribution --
COLOR_MAX = 3

# -- Empty cells / outcomes --
EMPTY_MIN, EMPTY_MAX = 2, 4
P_SUCCESS = 0.4
P_QUERY_4_BLOCKED = 0.55

# -- Generation --
MAX_RETRIES = 500
EMPTY_LAYOUT_RETRIES = 400
COLOR_ASSIGN_RETRIES = 200


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


def _choose_empty_cells() -> set[tuple[int, int]]:
    for _ in range(EMPTY_LAYOUT_RETRIES):
        count = random.randint(EMPTY_MIN, EMPTY_MAX)
        empty_cells = set(random.sample(list(ALL_CELLS), count))
        row_counts, col_counts = _line_empty_counts(empty_cells)
        counts = row_counts + col_counts
        if max(counts) >= TABLE_DIM:
            continue
        if not _open_lines(empty_cells):
            continue
        if not _blocked_lines(empty_cells):
            continue
        return empty_cells
    raise RuntimeError("failed to construct empty-cell layout with both open and blocked lines")


def _balanced_shape_masks(num_shapes: int) -> list[int]:
    counts = {k: 0 for k in range(FILL_MIN, FILL_MAX + 1)}
    fill_levels = list(counts)
    random.shuffle(fill_levels)
    total = 0
    for k in fill_levels:
        if total == num_shapes:
            break
        counts[k] += 1
        total += 1

    while total < num_shapes:
        choices = [
            k
            for k in range(FILL_MIN, FILL_MAX + 1)
            if counts[k] < min(MAX_SHAPES_PER_FILL, len(PATTERNS_BY_FILL[k]))
        ]
        counts[random.choice(choices)] += 1
        total += 1

    masks: list[int] = []
    for k, n in counts.items():
        if n:
            masks.extend(random.sample(PATTERNS_BY_FILL[k], n))
    random.shuffle(masks)
    return masks


def _balanced_color_counts(num_cells: int) -> dict[int, int]:
    counts = {c: 0 for c in GLYPH_COLORS}
    colors = list(GLYPH_COLORS)
    random.shuffle(colors)

    total = 0
    for color in colors:
        if total == num_cells:
            break
        counts[color] += 1
        total += 1

    while total < num_cells:
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
        total += 1

    return counts


def _assign_family_colors(families: list[list[tuple[int, int]]]) -> list[list[int]]:
    num_cells = sum(len(family) for family in families)
    for _ in range(COLOR_ASSIGN_RETRIES):
        remaining = _balanced_color_counts(num_cells)
        family_colors: list[list[int]] = []
        ok = True
        for family in sorted(enumerate(families), key=lambda item: len(item[1]), reverse=True):
            idx, cells = family
            if len(cells) == 2:
                color_a_choices = [c for c, n in remaining.items() if n > 0]
                if not color_a_choices:
                    ok = False
                    break
                color_a = random.choice(color_a_choices)
                remaining[color_a] -= 1
                color_b_choices = [c for c, n in remaining.items() if n > 0 and c != color_a]
                if not color_b_choices:
                    ok = False
                    break
                color_b = random.choice(color_b_choices)
                remaining[color_b] -= 1
                colors = [int(color_a), int(color_b)]
                random.shuffle(colors)
            else:
                single_choices = [c for c, n in remaining.items() if n > 0]
                if not single_choices:
                    ok = False
                    break
                color = random.choice(single_choices)
                remaining[color] -= 1
                colors = [int(color)]
            family_colors.append((idx, colors))
        if not ok:
            continue

        family_colors.sort(key=lambda item: item[0])
        return [colors for _idx, colors in family_colors]
    raise RuntimeError("failed to assign family colors")


def _partition_remaining_cells(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    shuffled = cells.copy()
    random.shuffle(shuffled)
    families: list[list[tuple[int, int]]] = []
    i = 0
    while i + 1 < len(shuffled):
        families.append([shuffled[i], shuffled[i + 1]])
        i += 2
    if i < len(shuffled):
        families.append([shuffled[i]])
    return families


def _query_sources_for_families(
    families: list[list[tuple[int, int]]],
    family_ids: list[int],
) -> list[tuple[int, int]]:
    exemplar_by_family = {
        family_id: random.choice(families[family_id]) for family_id in set(family_ids)
    }
    query_sources = [exemplar_by_family[family_id] for family_id in family_ids]
    random.shuffle(query_sources)
    return [(int(r), int(c)) for r, c in query_sources]


def _build_families(
    nonempty_cells: list[tuple[int, int]],
    target_cells: list[tuple[int, int]],
    success: bool,
) -> tuple[list[list[tuple[int, int]]], list[int], list[tuple[int, int]]]:
    target_set = set(target_cells)
    remainder = [cell for cell in nonempty_cells if cell not in target_set]
    families: list[list[tuple[int, int]]] = []
    query_family_ids: list[int]

    shuffled_target = target_cells.copy()
    random.shuffle(shuffled_target)
    if success:
        assert len(shuffled_target) == TABLE_DIM
        families.append([shuffled_target[0], shuffled_target[1]])
        families.append([shuffled_target[2], shuffled_target[3]])
        query_family_ids = [0, 0, 1, 1]
    else:
        assert len(shuffled_target) == TABLE_DIM - 1
        families.append([shuffled_target[0], shuffled_target[1]])
        families.append([shuffled_target[2]])
        query_family_ids = [0, 0, 1, 1] if random.random() < P_QUERY_4_BLOCKED else [0, 0, 1]

    families.extend(_partition_remaining_cells(remainder))
    matched_cells = sorted(target_cells)
    return families, query_family_ids, matched_cells


def _render_table(
    grid: np.ndarray,
    families: list[list[tuple[int, int]]],
) -> dict[tuple[int, int], np.ndarray]:
    family_masks = _balanced_shape_masks(len(families))
    family_colors = _assign_family_colors(families)
    glyphs_by_cell: dict[tuple[int, int], np.ndarray] = {}

    for family_cells, mask, colors in zip(families, family_masks, family_colors, strict=True):
        assert len(family_cells) == len(colors)
        for (r_i, c_i), color in zip(family_cells, colors, strict=True):
            glyph = _glyph(mask, color)
            glyphs_by_cell[(int(r_i), int(c_i))] = glyph
            paste(grid, glyph, ROWS[r_i], COLS[c_i])
    return glyphs_by_cell


def _expected_output(
    grid: np.ndarray,
    matched_cells: list[tuple[int, int]],
    success: bool,
) -> np.ndarray:
    out = grid.copy()
    for r_i, c_i in matched_cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell[cell == BG] = GREEN
    if success:
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == BG] = GREEN
        out[VERDICT_ROW:, :] = GREEN
    else:
        out[VERDICT_ROW:, :] = RED
    return out


def _exact_signature(cell: np.ndarray) -> tuple[tuple[int, int, int], ...] | None:
    mask = cell != BG
    if not bool(mask.any()):
        return None
    return tuple((int(r), int(c), int(cell[r, c])) for r, c in zip(*np.where(mask), strict=False))


def _cross_color_extra_matches(
    grid: np.ndarray,
    query_sources: list[tuple[int, int]],
    matched_cells: list[tuple[int, int]],
) -> int:
    query_exact = {
        _exact_signature(grid[ROWS[r] : ROWS[r] + CELL, COLS[c] : COLS[c] + CELL])
        for r, c in query_sources
    }
    matched_exact = [
        _exact_signature(grid[ROWS[r] : ROWS[r] + CELL, COLS[c] : COLS[c] + CELL])
        for r, c in matched_cells
    ]
    return sum(sig not in query_exact for sig in matched_exact)


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    empty_cells = _choose_empty_cells()
    nonempty_cells = [cell for cell in ALL_CELLS if cell not in empty_cells]
    open_lines = _open_lines(empty_cells)
    blocked_lines = _blocked_lines(empty_cells)
    assert open_lines and blocked_lines

    success = random.random() < P_SUCCESS
    target_cells = random.choice(open_lines if success else blocked_lines).copy()
    families, query_family_ids, matched_cells = _build_families(nonempty_cells, target_cells, success)
    glyphs_by_cell = _render_table(grid, families)
    query_sources = _query_sources_for_families(families, query_family_ids)

    for slot, cell in enumerate(query_sources):
        paste(grid, glyphs_by_cell[cell], 0, COLS[slot])

    assert _cross_color_extra_matches(grid, query_sources, matched_cells) > 0
    if not success:
        rows = {r for r, _c in matched_cells}
        cols = {c for _r, c in matched_cells}
        assert len(matched_cells) == TABLE_DIM - 1
        assert len(rows) == 1 or len(cols) == 1

    expected = _expected_output(grid, matched_cells, success)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo variant input")
