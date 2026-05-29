# Combo variant of d8e07eb2 (index 26)
# combo_spec: D2/3+D3/1+D5/1
# Name: 干扰查询+空格参考表+2x2判定

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis
# a) 具体化与行为追踪
# - 画面仍是 30x22：顶带 5 行、两条整行分隔线、中央 4x4 表格、底部 2 行判定带。
# - 但顶带不再是四个 5x5 查询格。由于固定宽度不足以容纳“4 个真实查询 + 1~2 个干扰项”的
#   五个 5x5 槽位，查询区压缩为五个 3x3 槽位，位于第 1 行起、列 {1,5,9,13,17}。
# - 中央参考表仍是 4x4 的 5x5 单元，但有 2~4 个单元完全为空。非空单元中的字形仍是居中的
#   3x3 单色图案，因此查询与表格匹配时必须把 5x5 单元和裸 3x3 查询都归一化到同一局部签名。
# - 成功条件改为：过滤掉顶带中在表格里不存在的干扰字形后，剩下的真实匹配格必须恰好构成一个
#   连续 2x2 子矩阵。若某个 2x2 中有空单元，则查询最多只能得到其中 3 个真实字形，即使另外
#   两维都“看起来接近成功”，仍只能判红。
# - 具体例子：顶带显示 [A, X, B, C, D]，其中 X 是干扰；表中 A/B/C/D 正好位于连续 2x2，
#   则输出高亮这四格并将顶带背景与底带涂绿。若顶带显示 [A, X, B, C, Y]，而 A/B/C 是某个
#   2x2 的三格、第四格为空，则只高亮三格，顶带保持原样，底带变红。
#
# b) 复杂度校准
# - 干扰项固定为：成功样例 1 个，失败样例 1 或 2 个。这样“先筛掉无匹配项”会稳定出现。
# - 空单元固定为 2~4 个，并要求同时存在：
#   1) 至少一个完全非空的 2x2，可用于成功样例；
#   2) 至少一个恰含 1 个空位的 2x2，可用于“几乎凑成方块但被空格打断”的失败样例。
# - 非空表格单元的 (颜色, 形状) 仍保持唯一，避免筛干扰与查表时出现多重匹配。
#
# c) 隐式假设审查
# - 原题默认“所有可见查询都是有效查询”，D2/3 破坏了这一点，因此 solver 必须先构建表格签名集，
#   再把顶带签名过滤成真实查询集。
# - 原题通过首个前景像素反推表格起点；D3/1 下首行首列可能全空，这个推断不再可靠，因此改为
#   根据分隔线和固定 4x4/5x5 几何直接计算表格锚点。
# - 原题把“4 个匹配且同一行/列”当成功；D5/1 下必须改成“4 个匹配且恰是连续 2x2”。
# - 新 solver 还额外假设：顶带查询槽位固定为五个 3x3 窗口，真实查询彼此不同，干扰签名不出现在
#   表格中。generator 会显式保证这些条件。
#
# d) 放置合法性
# - 查询槽位互不重叠，全部落在 rows 1..3 与 cols 1..19 内。
# - 表格仍使用原题的 5x5 晶格：左上角 (7,1)，四行四列无越界；空单元仅表示不粘贴字形。
# - 生成器只从非空表格单元复制真实查询，因此不会产生“查询引用空单元”的非法输入。
#
# e) 视觉可观察性
# - 人眼能直接看到：顶带有 5 个小字形，但输出不会对每个都响应；表中还有明显空洞；成功高亮不再
#   是直线，而是紧凑的 2x2 方块。
# - 失败样例中会稳定出现“3 格 L 形 + 空缺角”的情况，能把“空单元阻断 2x2”这条规则暴露出来。
#
# f) 可视化整洁度
# - 顶带只放一层 3x3 字形，表格只放一层居中字形，输出叠加仍只有绿色背景高亮与底带判定色。
# - 空单元保持纯背景，不与字形叠加；干扰项只存在于顶带，不会与表格高亮混淆。
#
# g) 变化的可推断性
# - 没有引入任何基于颜色编号的隐式排序。颜色只用于和形状共同标识字形。
# - 新规律完全来自可见结构：顶带多出的无匹配字形、表格中的空缺位置、以及匹配格是否形成连续 2x2。


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
BLOCK_SIDE = 2
BLOCK_AREA = BLOCK_SIDE * BLOCK_SIDE

# -- Query geometry --
QUERY_SIZE = 3
QUERY_ROW = 1
QUERY_COLS = (1, 5, 9, 13, 17)
TOTAL_QUERY_SLOTS = len(QUERY_COLS)

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5

# -- Empty cells / outcomes --
EMPTY_MIN, EMPTY_MAX = 2, 4
P_SUCCESS = 0.35
P_BLOCKED_FAIL = 0.55
P_FOUR_REAL_FAIL = 0.6

# -- Generation --
MAX_RETRIES = 500
EMPTY_LAYOUT_RETRIES = 400
FAIL_SAMPLE_RETRIES = 400


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _normalized_signature(patch: np.ndarray, bg: int) -> tuple[tuple[int, int, int], ...] | None:
    mask = patch != int(bg)
    if not bool(mask.any()):
        return None
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    return tuple((int(r - r0), int(c - c0), int(patch[r, c])) for r, c in coords)


def _cell_signature(
    g: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
) -> tuple[tuple[int, int, int], ...] | None:
    patch = g[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)]
    return _normalized_signature(patch, bg)


def _table_starts(bottom_sep: int, width: int) -> tuple[list[int], list[int]]:
    row0 = int(bottom_sep) - TABLE_DIM * CELL
    col0 = (int(width) - TABLE_DIM * CELL) // 2
    row_starts = [row0 + i * CELL for i in range(TABLE_DIM)]
    col_starts = [col0 + i * CELL for i in range(TABLE_DIM)]
    return row_starts, col_starts


def _forms_2x2_block(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    cell_set = {(int(r), int(c)) for r, c in cells}
    if len(cell_set) != BLOCK_AREA:
        return False
    rows = sorted({r for r, _c in cell_set})
    cols = sorted({c for _r, c in cell_set})
    return (
        len(rows) == BLOCK_SIDE
        and len(cols) == BLOCK_SIDE
        and rows[1] == rows[0] + 1
        and cols[1] == cols[0] + 1
    )


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
ALL_QUERY_PATTERNS = tuple(mask for masks in PATTERNS_BY_FILL.values() for mask in masks)


def _balanced_color_list(num_cells: int) -> list[int]:
    assert num_cells >= len(GLYPH_COLORS), "not enough non-empty cells for color minimums"
    counts = {c: 1 for c in GLYPH_COLORS}
    extras = num_cells - len(GLYPH_COLORS)
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < 3]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _balanced_shape_masks(num_cells: int) -> list[int]:
    assert num_cells >= sum(SHAPE_BASE_COUNTS.values()), "not enough non-empty cells for shape minimums"
    counts = dict(SHAPE_BASE_COUNTS)
    extras = num_cells - sum(counts.values())
    for _ in range(extras):
        choices = [
            k
            for k in range(FILL_MIN, FILL_MAX + 1)
            if counts[k] < min(MAX_SHAPES_PER_FILL, len(PATTERNS_BY_FILL[k]))
        ]
        counts[random.choice(choices)] += 1

    masks: list[int] = []
    for fill, n in counts.items():
        masks.extend(random.sample(PATTERNS_BY_FILL[fill], n))
    random.shuffle(masks)
    return masks


def _table_glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _query_glyph(mask: int, color: int) -> np.ndarray:
    glyph = np.full((QUERY_SIZE, QUERY_SIZE), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    glyph[bits] = color
    return glyph


def _signature_from_mask(mask: int, color: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(r), int(c), int(color))
        for r in range(3)
        for c in range(3)
        if (mask >> (r * 3 + c)) & 1
    )


def _open_blocks(empty_cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    blocks: list[list[tuple[int, int]]] = []
    for top in range(TABLE_DIM - 1):
        for left in range(TABLE_DIM - 1):
            cells = [(top + dr, left + dc) for dr in range(BLOCK_SIDE) for dc in range(BLOCK_SIDE)]
            if all(cell not in empty_cells for cell in cells):
                blocks.append(cells)
    return blocks


def _blocked_blocks(empty_cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    blocks: list[list[tuple[int, int]]] = []
    for top in range(TABLE_DIM - 1):
        for left in range(TABLE_DIM - 1):
            cells = [(top + dr, left + dc) for dr in range(BLOCK_SIDE) for dc in range(BLOCK_SIDE)]
            if sum(cell in empty_cells for cell in cells) == 1:
                blocks.append([cell for cell in cells if cell not in empty_cells])
    return blocks


def _choose_empty_cells() -> set[tuple[int, int]]:
    for _ in range(EMPTY_LAYOUT_RETRIES):
        count = random.randint(EMPTY_MIN, EMPTY_MAX)
        empty_cells = set(random.sample(list(ALL_CELLS), count))
        if _open_blocks(empty_cells) and _blocked_blocks(empty_cells):
            return empty_cells
    raise RuntimeError("failed to place empty cells with both open and blocked 2x2 blocks")


def _sample_fail_cells(
    nonempty_cells: list[tuple[int, int]],
    real_count: int,
) -> list[tuple[int, int]]:
    for _ in range(FAIL_SAMPLE_RETRIES):
        cells = random.sample(nonempty_cells, real_count)
        if _forms_2x2_block(cells):
            continue
        random.shuffle(cells)
        return cells
    raise RuntimeError("failed to sample non-success query cells")


def _pick_distractors(
    table_masks: list[int],
    used_query_sigs: set[tuple[tuple[int, int, int], ...]],
    count: int,
) -> list[tuple[int, int]]:
    distractors: list[tuple[int, int]] = []
    absent_masks = [mask for mask in ALL_QUERY_PATTERNS if mask not in set(table_masks)]
    while len(distractors) < count:
        mask = random.choice(absent_masks)
        color = random.choice(GLYPH_COLORS)
        sig = _signature_from_mask(mask, color)
        if sig in used_query_sigs:
            continue
        used_query_sigs.add(sig)
        distractors.append((mask, color))
    return distractors


def _expected_output(grid: np.ndarray, real_cells: list[tuple[int, int]], success: bool) -> np.ndarray:
    out = grid.copy()
    for r_i, c_i in real_cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell[cell == BG] = GREEN
    if success:
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == BG] = GREEN
        out[VERDICT_ROW:, :] = GREEN
    else:
        out[VERDICT_ROW:, :] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match only the real query glyphs, skip blank table cells, and test for a 2x2 block.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - The top band contains five compact 3x3 query glyph slots; 3-4 are genuine
      copies from the table and the remaining 1-2 are distractors with no match.
    - Between separators is a centered 4x4 table of 5x5 cells; some cells hold a
      centered 3x3 glyph while other cells are completely empty teal.

    Rule:
    - Build exact color+shape signatures for every non-empty table cell, then scan
      the five top slots and keep only the query glyphs whose signatures appear in
      the table at all.
    - Green-fill the background pixels of each matched table cell.
    - If the surviving matched cells are exactly four cells and they occupy one
      contiguous 2x2 submatrix of the table, also green-fill the query-band
      background and the verdict band; otherwise leave the query band unchanged and
      fill the verdict band red.

    Key insight: solving now requires three filters in sequence: ignore top-band
    distractors, ignore blank reference cells because they have no signature, and
    judge success by a compact 2x2 arrangement rather than any straight line.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    bg = get_bg(g)

    sep_rows = _constant_non_bg_rows(g, bg)
    assert len(sep_rows) >= 2, "expected two full-width separator rows"
    top_sep = sep_rows[0]
    bottom_sep = sep_rows[-1]

    row_starts, col_starts = _table_starts(bottom_sep, w)
    assert row_starts[0] > top_sep, "table must lie below top separator"
    assert row_starts[-1] + CELL <= bottom_sep, "table extends past bottom separator"
    assert col_starts[0] >= 0 and col_starts[-1] + CELL <= w, "table extends past grid width"

    table_map: dict[tuple[tuple[int, int, int], ...], tuple[int, int]] = {}
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c, CELL, CELL, bg)
            if sig is None:
                continue
            table_map[sig] = (int(ri), int(ci))
    assert table_map, "reference table must contain at least one glyph"

    query_sigs = [
        sig
        for c in QUERY_COLS
        if (sig := _cell_signature(g, QUERY_ROW, c, QUERY_SIZE, QUERY_SIZE, bg)) is not None
    ]
    real_query_sigs = {sig for sig in query_sigs if sig in table_map}

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for sig in real_query_sigs:
        ri, ci = table_map[sig]
        selected.append((ri, ci))
        cell = out[row_starts[ri] : row_starts[ri] + CELL, col_starts[ci] : col_starts[ci] + CELL]
        cell[cell == bg] = GREEN

    success = len(real_query_sigs) == BLOCK_AREA and len(selected) == BLOCK_AREA and _forms_2x2_block(selected)

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = GREEN
        out[bottom_sep + 1 :, :] = GREEN
    else:
        out[bottom_sep + 1 :, :] = RED

    return out


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    empty_cells = _choose_empty_cells()
    nonempty_cells = [cell for cell in ALL_CELLS if cell not in empty_cells]
    open_blocks = _open_blocks(empty_cells)
    blocked_blocks = _blocked_blocks(empty_cells)
    assert open_blocks and blocked_blocks

    masks = _balanced_shape_masks(len(nonempty_cells))
    colors = _balanced_color_list(len(nonempty_cells))
    glyph_specs: dict[tuple[int, int], tuple[int, int]] = {}
    table_masks: list[int] = []
    for cell, mask, color in zip(nonempty_cells, masks, colors, strict=True):
        glyph_specs[cell] = (mask, color)
        table_masks.append(mask)
        paste(grid, _table_glyph(mask, color), ROWS[cell[0]], COLS[cell[1]])

    success = random.random() < P_SUCCESS
    if success:
        real_cells = random.choice(open_blocks).copy()
    else:
        if random.random() < P_BLOCKED_FAIL:
            real_cells = random.choice(blocked_blocks).copy()
        else:
            real_count = 4 if random.random() < P_FOUR_REAL_FAIL else 3
            real_cells = _sample_fail_cells(nonempty_cells, real_count)

    used_query_sigs = {
        _signature_from_mask(glyph_specs[cell][0], glyph_specs[cell][1]) for cell in real_cells
    }
    distractor_count = TOTAL_QUERY_SLOTS - len(real_cells)
    distractors = _pick_distractors(table_masks, used_query_sigs, distractor_count)

    query_items: list[tuple[str, tuple[int, int]]] = [
        ("real", cell) for cell in real_cells
    ] + [("distractor", spec) for spec in distractors]
    random.shuffle(query_items)
    assert len(query_items) == TOTAL_QUERY_SLOTS

    for slot, (kind, payload) in enumerate(query_items):
        if kind == "real":
            cell = payload
            mask, color = glyph_specs[cell]
        else:
            mask, color = payload
        paste(grid, _query_glyph(mask, color), QUERY_ROW, QUERY_COLS[slot])

    expected = _expected_output(grid, real_cells, success)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
