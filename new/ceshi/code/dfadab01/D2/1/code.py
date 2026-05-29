# Variant of dfadab01
# Dimension: D2 — 标记色与字形色的映射关系 (transform)
# Change: 固定全局映射：标记色2->字形色4（空心方块），3->1（菱形），5->6（对角块），8->7（X形），映射关系跨所有示... → 样本字形定义局部映射

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ===============
# concrete_behavior:
#   - 保留原题的 4 个 4x4 glyph 库不变：黄色空心方块、蓝色菱形环、
#     洋红对角块、橙色 X。变化只在“哪种 marker 颜色触发哪一个 glyph”。
#   - 输入上半部放 2-4 个图例；每个图例由一个完整 4x4 glyph 和它右下角
#     (+4,+4) 的单像素 label marker 构成。label 的颜色就是要被解码的
#     marker 颜色，glyph 的形状+颜色给出该颜色在本题里的局部含义。
#   - 输入下半部仍是原题风格的规则点阵 marker。输出中图例整体消失，
#     只把真实 marker 展开成它们在本题图例里定义的 4x4 glyph。
#   - 具体追踪例子：若某题顶部出现“橙色 X，右下角 label=2”，原题会把
#     所有 `2` 展开成黄色方框；本变体则必须把所有真实 `2` 展开成橙色 X。
#     差异发生在 `marker_color -> (glyph_color, mask)` 的获取环节。
# complexity_calibration:
#   - glyph 库仍只有 4 类，形状与颜色都显著不同，复杂度与原题同量级。
#   - 每题使用 2-4 种 marker 颜色，且每种真实颜色至少重复 2 次，保证
#     人类和 solver 都能从少量样例中看到“图例解码后被多次复用”。
#   - 生成器强制至少 2 个颜色的映射偏离原题固定表，避免大部分样例退化成
#     原规则或只出现一处不明显变化。
# hidden_assumption_review:
#   - 原 `GLYPHS` 把 marker 颜色直接硬编码到 glyph；该前提在 D2 中失效。
#   - 原 `_has_source_glyph(... glyph_color, mask)` 只在已知全局映射时可用；
#     现在必须改成“对任意 glyph 库成员尝试匹配”，先识别 legend label。
#   - 原主循环中“边扫描 marker 边决定是否跳过 legend label”依赖已知类型；
#     现在必须分两段：先收集 legend 映射，再绘制非 legend 的真实 marker。
# placement_legality:
#   - 图例固定在顶行，top-left 列槽位为 0/5/10/15；这样 glyph 占 4 列、
#     label 位于第 5 列，多个图例彼此不重叠且全部在 20x20 网格内。
#   - 真实 marker 网格从第 6 行开始，行列间距均为 4 或 5，因此展开后的
#     4x4 glyph 不会互相重叠；最末行/列可贴边，从而保留原题的裁剪现象。
#   - 真实 marker 永不放到图例区，也不放到任何 label 单元，防止 solver
#     把普通 marker 误判成 legend label。
# visual_observability:
#   - 顶部是一排显式图例，底部是一片待展开 marker，分区明确，读图时能先
#     看“字典”再看应用对象，D2 的局部映射变化在视觉上直接可见。
# tidiness:
#   - 图例和真实 marker 分带摆放，没有对象叠压；输出只保留展开结果，画面
#     仍然整洁，和原题一样由离散 4x4 模块构成。
# color_inferability:
#   - 只有 4 个 glyph 颜色，且每个颜色总伴随固定轮廓出现；解题者并不需要
#     记数字 id，只需从图例观察“哪个 label 对应哪个彩色轮廓”。
# new_main_assumptions_to_check:
#   - 每个在真实区域出现的 marker 颜色都必须至少有一个 legend label。
#   - legend label 是唯一一种“其左上 4x4 正好是某个 glyph 样本”的 marker。
#   - glyph 库颜色仍互异，因此 4x4 样本不会匹配到多个 glyph 类型。
#   - generate_input() 通过顶部独立图例区和完整 legend 覆盖来保证这些前提。


MARKER_COLORS: tuple[int, ...] = (2, 3, 5, 8)
BASE_GLYPH_BY_MARKER: dict[int, int] = {2: 0, 3: 1, 5: 2, 8: 3}
LEGEND_TOP = 0
LEGEND_LABEL_ROW = 4
LEGEND_SLOTS: tuple[int, ...] = (0, 5, 10, 15)

GLYPH_LIBRARY: tuple[tuple[int, np.ndarray], ...] = (
    (
        4,
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    (
        1,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    (
        6,
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    (
        7,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
)
GLYPH_COLORS = {glyph_color for glyph_color, _ in GLYPH_LIBRARY}
ALLOWED_COLORS = {0, *MARKER_COLORS, *GLYPH_COLORS}


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _decode_legend_mapping(
    g: np.ndarray,
) -> tuple[dict[int, tuple[int, np.ndarray]], set[tuple[int, int]]]:
    mapping: dict[int, tuple[int, np.ndarray]] = {}
    legend_labels: set[tuple[int, int]] = set()

    for marker_color in MARKER_COLORS:
        coords = np.argwhere(g == marker_color)
        for r, c in coords:
            r_i = int(r)
            c_i = int(c)
            matches: list[int] = []
            for glyph_idx, (glyph_color, mask) in enumerate(GLYPH_LIBRARY):
                if _has_source_glyph(g, r_i - 4, c_i - 4, glyph_color, mask):
                    matches.append(glyph_idx)
            if not matches:
                continue
            assert len(matches) == 1, f"ambiguous legend for marker color {marker_color} at {(r_i, c_i)}"
            glyph_color, mask = GLYPH_LIBRARY[matches[0]]
            if marker_color in mapping:
                prev_color, prev_mask = mapping[marker_color]
                assert prev_color == glyph_color and np.array_equal(prev_mask, mask)
            mapping[marker_color] = (glyph_color, mask)
            legend_labels.add((r_i, c_i))

    return mapping, legend_labels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored control pixel using the per-puzzle legend-defined 4x4 glyph mapping.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - The top band contains 2-4 legend entries: a sample 4x4 glyph plus a same-rule label marker one cell below-right of it.

    Rule:
    - First decode each legend: the label marker color tells which marker type owns the sample glyph immediately up-left of it.
    - Replace every non-legend marker of that color by the decoded glyph, placing the glyph's top-left corner at the marker and clipping at the grid edge.
    - Remove the legend glyphs themselves and skip the legend label markers in the output.

    Key insight:
    - The glyph alphabet is still fixed, but the marker-to-glyph assignment is local to each puzzle and must be read from the visible legend before expanding the real markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset(ALLOWED_COLORS)

    mapping, legend_labels = _decode_legend_mapping(g)
    present_marker_colors = {int(color) for color in np.unique(g) if int(color) in MARKER_COLORS}
    assert present_marker_colors, "expected at least one marker color"
    for marker_color in present_marker_colors:
        assert marker_color in mapping, f"missing legend for marker color {marker_color}"

    out = np.zeros_like(g)
    for marker_color in present_marker_colors:
        glyph_color, mask = mapping[marker_color]
        for r, c in np.argwhere(g == marker_color):
            cell = (int(r), int(c))
            if cell in legend_labels:
                continue
            _paint_clipped(out, cell[0], cell[1], glyph_color, mask)

    return out


def _sample_real_mapping(used_colors: list[int]) -> dict[int, int]:
    min_moved = 2 if len(used_colors) >= 2 else 1
    while True:
        glyph_ids = random.sample(range(len(GLYPH_LIBRARY)), len(used_colors))
        mapping = {marker_color: glyph_id for marker_color, glyph_id in zip(used_colors, glyph_ids)}
        moved = sum(int(mapping[color] != BASE_GLYPH_BY_MARKER[color]) for color in used_colors)
        if moved >= min_moved:
            return mapping


def _sample_lattice(h: int, w: int, used_count: int) -> tuple[list[int], list[int]]:
    while True:
        rows = random.randint(2, 4)
        cols = random.randint(3, 5)
        if rows * cols < 2 * used_count:
            continue
        row_spacing = random.randint(4, 5)
        col_spacing = random.randint(4, 5)
        max_row_start = h - 1 - row_spacing * (rows - 1)
        max_col_start = w - 1 - col_spacing * (cols - 1)
        if max_row_start < 6 or max_col_start < 0:
            continue
        start_r = random.randint(6, max_row_start)
        start_c = random.randint(0, max_col_start)
        row_positions = [start_r + i * row_spacing for i in range(rows)]
        col_positions = [start_c + i * col_spacing for i in range(cols)]
        return row_positions, col_positions


def _build_case() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    h = 20
    w = 20

    used_count = random.choices([2, 3, 4], weights=[2, 3, 3], k=1)[0]
    used_colors = random.sample(list(MARKER_COLORS), used_count)
    mapping = _sample_real_mapping(used_colors)
    row_positions, col_positions = _sample_lattice(h, w, used_count)

    cells = [(r, c) for r in row_positions for c in col_positions]
    total_cells = len(cells)
    min_fill = max((3 * total_cells + 4) // 5, 2 * used_count)
    fill_count = random.randint(min_fill, total_cells)
    chosen_cells = random.sample(cells, fill_count)

    real_marker_colors = [color for color in used_colors for _ in range(2)]
    real_marker_colors.extend(random.choices(used_colors, k=fill_count - len(real_marker_colors)))
    random.shuffle(real_marker_colors)

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    legend_slots = random.sample(list(LEGEND_SLOTS), used_count)
    random.shuffle(legend_slots)
    for marker_color, legend_col in zip(used_colors, legend_slots):
        glyph_color, mask = GLYPH_LIBRARY[mapping[marker_color]]
        _paint_clipped(inp, LEGEND_TOP, legend_col, glyph_color, mask)
        inp[LEGEND_LABEL_ROW, legend_col + 4] = marker_color

    for (r, c), marker_color in zip(chosen_cells, real_marker_colors):
        inp[r, c] = marker_color
        glyph_color, mask = GLYPH_LIBRARY[mapping[marker_color]]
        _paint_clipped(out, r, c, glyph_color, mask)

    solved = main(inp)
    assert np.array_equal(solved, out)

    meta = {
        "used_colors": tuple(sorted(used_colors)),
        "mapping": {color: mapping[color] for color in sorted(used_colors)},
        "moved_count": sum(int(mapping[color] != BASE_GLYPH_BY_MARKER[color]) for color in used_colors),
        "legend_count": used_count,
        "real_marker_count": fill_count,
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    inp, _, _ = _build_case()
    return inp


def generate_pair() -> tuple[np.ndarray, np.ndarray]:
    inp, out, _ = _build_case()
    return inp, out
