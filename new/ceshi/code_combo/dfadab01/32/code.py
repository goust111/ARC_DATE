# Combo variant of dfadab01 (index 32)
# combo_spec: D2/0+D4/0
# Name: 标记色保持+样本保留

from __future__ import annotations

import random

import numpy as np


# Design analysis
# - a_concretization_and_trace:
#   - 原题里，输入中的单像素 marker 会被替换成固定 4x4 字形，但输出色和 marker 色不同，
#     同时 legend glyph 与其右下标签都会消失。
#   - 本融合变体里，几何字典仍然固定不变，但输出色改为“等于 marker 自身颜色”；
#     同时若输入里出现一个完整 legend glyph，它会原样保留在输出里，只删除其右下标签。
#   - 具体例子：若输入左侧出现一个颜色 3 的菱形样本，右下角有一个颜色 3 的单像素标签，
#     且主体区还有若干颜色 3 的单像素 marker，那么：
#     原题输出会删掉左侧样本，并把主体区 marker 画成颜色 1 的菱形；
#     本题输出会保留左侧颜色 3 的样本，并把主体区 marker 画成颜色 3 的菱形。
#   - 差异点有两处并且相互作用：一是颜色不再通过 2->4、3->1 之类的查表改变，
#     二是样本与新实例现在使用同一颜色体系，因此输出里能同时看到“原件”和“复制品”。
# - b_complexity_calibration:
#   - 若不强制生成 legend，本组合会退化成只体现 D2/0；因此生成器固定放 1 个 legend。
#   - 若只用 1 种 marker 色，人类较难从少量样例中确认“颜色保持”是一般规律；
#     因此主体区至少使用 2 种颜色，并让每种颜色重复出现。
#   - 几何仍限制为 4 个固定 4x4 图案，避免自由度过多导致图例和真实实例都像噪声。
# - c_implicit_assumption_audit:
#   - 原 main() 中 `GLYPHS` 的第一个元素是 glyph_color；这里这个前提失效，颜色不再独立存储，
#     而是直接等于 marker_color。
#   - 原 `_has_source_glyph()` 只需检查“左上 4x4 是否是输出色 glyph”；这里颜色相同了，
#     仍可用同一思路识别 legend label，但必须更严格地要求 glyph 补丁的非 mask 区为背景，
#     否则可能把别的同色结构误判为 legend。
#   - 原 `out = np.zeros_like(g)` 后只扩展真实 marker，因为 legend 会消失；
#     这里不能直接复制输入，也不能只保留扩展结果。若直接 `out = g.copy()`，像颜色 3 的菱形
#     这类 top-left 为 0 的 glyph 会错误保留原 marker 像素；因此必须单独重画 legend glyph，
#     再单独扩展真实 marker，标签不写回。
# - d_placement_legality:
#   - 生成器固定使用 20x20，legend 放在最左列的一个 5x5 槽位内，主体 marker 放在第 8 列以后，
#     因而 legend 与主体区不重叠。
#   - 主体 marker 仍在间距 >= 4 的规则格点上，保证 4x4 glyph 之间不会互相覆盖；
#     末行/末列可落在 17 或 18，保留原题的边界裁剪现象。
# - e_visual_observability:
#   - 输入里直接可见一个完整样本；输出里它继续存在，而真实 marker 也展开成相同颜色的同形字形，
#     人类能从 2-4 个样例中直接观察到“样本保留 + 颜色不变”这两个现象。
#   - 至少两种颜色重复出现，使解题者看到多个“marker 色 == 输出字形色”的重复证据。
# - f_visual_cleanliness:
#   - legend 独立在左侧，主体区独立在右侧，没有多层覆盖；输出也仍是干净的黑底模块排布。
#   - 不让主体 marker 进入 legend 的 4x4 框和标签位，避免同色结构混杂。
# - g_color_inferability:
#   - 这里只有 4 种离散颜色类别，没有额外数值计算规则；“颜色保持”是最简单且可见的颜色规律。
#   - 因为 legend 和实例颜色完全一致，颜色变化线索不会被其他映射噪声掩盖。
# - new_main_assumptions_to_check:
#   - solver 额外假设：若某个 marker 像素的左上 4x4 恰好是同色固定 glyph，且补丁其它格为背景，
#     那么该 marker 是 legend label 而不是真实 marker。
#   - generate_input() 通过专用 legend 区和专用主体区保证这个假设成立。


GLYPHS: dict[int, np.ndarray] = {
    2: np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    3: np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    ),
    5: np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    ),
    8: np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=bool,
    ),
}

MARKER_COLORS: tuple[int, ...] = tuple(GLYPHS)
GLYPH_SIZE = 4
GRID_SIZE = 20
LEGEND_COL = 0
LEGEND_ROW_STARTS: tuple[int, ...] = (0, 5, 10, 15)
BODY_ROW_LAYOUTS: tuple[tuple[int, ...], ...] = (
    (0, 4, 8, 12, 16),
    (1, 5, 9, 13, 17),
    (2, 6, 10, 14, 18),
    (0, 5, 10, 15),
    (2, 7, 12, 17),
)
BODY_COL_LAYOUTS: tuple[tuple[int, ...], ...] = (
    (8, 12, 16),
    (9, 13, 17),
    (10, 14, 18),
    (8, 13, 18),
)
MAX_GEN_ATTEMPTS = 400


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + GLYPH_SIZE > h or c0 + GLYPH_SIZE > w:
        return False
    patch = g[r0 : r0 + GLYPH_SIZE, c0 : c0 + GLYPH_SIZE]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] == 0))


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(GLYPH_SIZE):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(GLYPH_SIZE):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _collect_legends(
    g: np.ndarray,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], list[tuple[int, int, int, np.ndarray]]]:
    legend_labels: set[tuple[int, int]] = set()
    legend_source_cells: set[tuple[int, int]] = set()
    legend_glyphs: list[tuple[int, int, int, np.ndarray]] = []
    for marker_color, mask in GLYPHS.items():
        for r, c in np.argwhere(g == marker_color):
            r_i = int(r)
            c_i = int(c)
            if _has_source_glyph(g, r_i - GLYPH_SIZE, c_i - GLYPH_SIZE, marker_color, mask):
                legend_labels.add((r_i, c_i))
                legend_glyphs.append((r_i - GLYPH_SIZE, c_i - GLYPH_SIZE, marker_color, mask))
                for dr in range(GLYPH_SIZE):
                    for dc in range(GLYPH_SIZE):
                        if bool(mask[dr, dc]):
                            legend_source_cells.add((r_i - GLYPH_SIZE + dr, c_i - GLYPH_SIZE + dc))
    return legend_labels, legend_source_cells, legend_glyphs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each control pixel with its fixed 4x4 glyph while keeping same-color sample glyphs intact.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Some inputs also include one full 4x4 sample glyph drawn in the same color as its marker type.
    - That sample has a same-color label pixel one cell below-right of its 4x4 bounding box.

    Rule:
    - Each real marker expands to its fixed 4x4 glyph, anchored with the marker at the glyph's top-left corner.
    - The glyph uses the marker's own color instead of a different mapped output color, and clipping at the grid edge still applies.
    - If a complete sample glyph is present, preserve that 4x4 sample exactly as-is in the output, but remove its below-right label pixel.

    Key insight:
    - The four shapes are still identified by marker color, but the color transformation disappears: samples and generated instances now share the same color system, and only the label pixel is suppressed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset({0, *MARKER_COLORS})

    legend_labels, legend_source_cells, legend_glyphs = _collect_legends(g)

    out = np.zeros_like(g)
    for r0, c0, marker_color, mask in legend_glyphs:
        _paint_clipped(out, r0, c0, marker_color, mask)

    seen_real_marker = False
    for marker_color, mask in GLYPHS.items():
        for r, c in np.argwhere(g == marker_color):
            rc = (int(r), int(c))
            if rc in legend_labels or rc in legend_source_cells:
                continue
            seen_real_marker = True
            _paint_clipped(out, rc[0], rc[1], marker_color, mask)

    assert seen_real_marker
    return out


def _body_positions() -> list[tuple[int, int]]:
    rows = list(random.choice(BODY_ROW_LAYOUTS))
    cols = list(random.choice(BODY_COL_LAYOUTS))
    positions = [(int(r), int(c)) for r in rows for c in cols]
    random.shuffle(positions)
    min_keep = max(8, int(np.ceil(0.6 * len(positions))))
    keep = random.randint(min_keep, len(positions))
    return positions[:keep]


def _sample_real_markers() -> list[tuple[int, int, int]]:
    positions = _body_positions()
    used_count = random.choices([2, 3, 4], weights=[2, 3, 3], k=1)[0]
    used_colors = random.sample(list(MARKER_COLORS), used_count)

    marker_colors = [color for color in used_colors for _ in range(2)]
    marker_colors.extend(random.choices(used_colors, k=len(positions) - len(marker_colors)))
    random.shuffle(marker_colors)

    return [(r, c, marker_colors[i]) for i, (r, c) in enumerate(positions)]


def _sample_legend_color(real_markers: list[tuple[int, int, int]]) -> int:
    used_colors = sorted({marker_color for _, _, marker_color in real_markers})
    return random.choice(used_colors)


def _build_output(legend_row: int, legend_color: int, real_markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    _paint_clipped(out, legend_row, LEGEND_COL, legend_color, GLYPHS[legend_color])
    for r, c, marker_color in real_markers:
        _paint_clipped(out, r, c, marker_color, GLYPHS[marker_color])
    return out


def _build_input(legend_row: int, legend_color: int, real_markers: list[tuple[int, int, int]]) -> np.ndarray:
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    _paint_clipped(g, legend_row, LEGEND_COL, legend_color, GLYPHS[legend_color])
    g[legend_row + GLYPH_SIZE, LEGEND_COL + GLYPH_SIZE] = legend_color
    for r, c, marker_color in real_markers:
        g[r, c] = marker_color
    return g


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        real_markers = _sample_real_markers()
        legend_row = random.choice(LEGEND_ROW_STARTS)
        legend_color = _sample_legend_color(real_markers)

        expected = _build_output(legend_row, legend_color, real_markers)
        inp = _build_input(legend_row, legend_color, real_markers)

        try:
            actual = main(inp)
        except Exception:
            continue

        if np.array_equal(actual, expected):
            return inp

    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
