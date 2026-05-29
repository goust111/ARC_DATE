# Combo variant of dfadab01 (index 30)
# combo_spec: D1/0+D6/0
# Name: 放大字形+网格自动扩展

from __future__ import annotations

import random

import numpy as np


"""
design_analysis:
  a_concretization_and_trace:
    original_behavior:
      - 每个真实标记都是单像素，输出把它替换成固定 4x4 字形，超出输入网格边界的部分直接裁剪。
      - 若输入里出现一个样本字形，则它只是图例；与该字形右下外侧一格的同色标记是标签，不参与扩写。
    fused_behavior:
      - 四种字形仍保持原题的固定字典与固定颜色映射，但整题共享一个统一放大倍数 k，基础 4x4 mask 通过像素复制放大到 8x8 或 12x12。
      - 标签偏移也随放大后的边界框变化：若样本左上角为 (r0, c0)，则标签位于 (r0 + 4k, c0 + 4k)。
      - 输出不再裁剪，而是把网格高度和宽度分别扩到足以容纳最靠下、最靠右的真实字形。
    concrete_trace:
      - 例如输入高宽为 19x19，统一 k=2，颜色 5 的真实标记在 (18, 18)。原题会尝试在 [18:22, 18:22] 画 4x4 并裁掉大半。
      - 融合后同一位置要画 8x8 的放大对角块，输出会扩到至少 26x26，使整块图案完整保留。
      - 若左上角还放了一个 8x8 的黄色样本，且其标签在 (8, 8)，则该标签像素要被识别为图例标签而不是新的真实标记。
  b_complexity_calibration:
    - 放大倍数只取离散集合 {2, 3}，分别得到 8x8 与 12x12，视觉上明显大于原题 4x4，同时自由度不至于失控。
    - 每个输入都保留原题的规则网格式标记分布，避免再叠加随机散布导致空间关系过乱。
    - 每个输入至少使用两种真实标记色，并且每种已用颜色至少出现两次，让解题者能从重复实例中确认“同色 -> 同字形、同尺寸”。
    - generator 始终放入至少一个 legend 样本，使 k 可以从输入内的可见结构直接读出，而不是依赖隐藏常量。
  c_implicit_assumption_review_of_main:
    - 原 _has_source_glyph() 写死 4x4 bbox；融合后必须改为读取当前放大 mask 的真实尺寸。
    - 原标签检测写死 (r - 4, c - 4)；融合后必须使用放大后边界框大小 size = 4k。
    - 原输出用 np.zeros_like(g) 固定输入尺寸；融合后必须先收集所有真实标记，再按最远右下角决定 out.shape。
    - 原 _paint_clipped() 依赖“允许裁剪”；融合后虽然仍可安全写成边界检查版本，但理论上 out 已经扩足，不应再发生信息丢失。
    - 新的 main() 额外假设：输入中至少有一个 legend，且所有 legend 使用同一个放大倍数。generate_input() 会构造性保证这点。
  d_placement_legality:
    - 真实标记仍放在规则网格上，间距至少为放大后字形大小再加 1，因此输出中的大字形不会互相重叠。
    - legend 固定放在左上保留区，真实标记网格整体压到右下方，避免 legend 样本、标签、真实标记三者混叠。
    - 真实标记网格的最后一行和最后一列都贴近输入下边或右边，确保“自动扩展”在绝大多数样本中显性触发，而不是退化成原题大小。
  e_visual_observability:
    - 放大后的字形与原 4x4 外观保持同构，但像素块更粗，人眼可直接看出“不是原题尺寸”。
    - 输出尺寸显著大于输入尺寸是可直接观察的显性结果，不依赖颜色编号或其他不可视信息。
    - legend 的样本边界框大小与标签偏移同步变化，因此解题者能从样本直接读出统一放大尺度。
  f_visual_tidiness:
    - legend 区域和真实标记区域分离，输入不会出现“样本像是在和真实标记重叠”的歧义。
    - 输出中的大字形不重叠，只是整体延展到更大的黑底网格，因此每个对象的来源清晰。
  g_change_inferability:
    - 没有新增颜色规则，颜色仍沿用原题固定映射；新增信息完全由几何尺度和输出尺寸承载。
    - 同一张图里会重复出现相同颜色的多个放大字形，足以支撑“统一放大”而不是“个别对象变大”的推断。
  h_behavior_metric:
    - 核心指标 1: expansion_delta = (output_h - input_h, output_w - input_w)，要求生成样本中二者都经常大于 0。
    - 核心指标 2: edge_trigger_count = 真实标记中满足 r + size > input_h 或 c + size > input_w 的个数；若该值长期为 0，则说明自动扩展并未真正被触发。
  i_generator_guarantees_for_new_main:
    - 至少一个 legend，且其标签严格位于对应放大样本的 (+size, +size)。
    - 至少一个真实标记落在输入最下最右的锚点上，因此 edge_trigger_count >= 1。
    - 所有真实标记使用与 legend 相同的统一 k，且输出色只出现在 legend 样本中，保证标签判定无歧义。
"""


BASE_GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    2: (
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
    3: (
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
    5: (
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
    8: (
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
}

MARKER_COLORS = tuple(BASE_GLYPHS.keys())
OUTPUT_COLORS = {glyph_color for glyph_color, _mask in BASE_GLYPHS.values()}
SCALE_FACTORS = (2, 3)
GEN_RETRIES = 300


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)


def _glyph_for_scale(marker_color: int, factor: int) -> tuple[int, np.ndarray]:
    glyph_color, base_mask = BASE_GLYPHS[marker_color]
    return glyph_color, _scale_mask(base_mask, factor)


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    gh, gw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + gh > h or c0 + gw > w:
        return False
    patch = g[r0 : r0 + gh, c0 : c0 + gw]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    gh, gw = mask.shape
    for dr in range(gh):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(gw):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _detect_scale(g: np.ndarray) -> int:
    matches: set[int] = set()
    for marker_color in MARKER_COLORS:
        coords = np.argwhere(g == marker_color)
        for r, c in coords:
            for factor in SCALE_FACTORS:
                glyph_color, mask = _glyph_for_scale(marker_color, factor)
                if _has_source_glyph(g, int(r) - mask.shape[0], int(c) - mask.shape[1], glyph_color, mask):
                    matches.add(int(factor))
    assert len(matches) == 1
    return int(next(iter(matches)))


def _is_legend_label(g: np.ndarray, r: int, c: int, marker_color: int, factor: int) -> bool:
    glyph_color, mask = _glyph_for_scale(marker_color, factor)
    return _has_source_glyph(g, r - mask.shape[0], c - mask.shape[1], glyph_color, mask)


def _collect_real_markers(g: np.ndarray, factor: int) -> list[tuple[int, int, int, np.ndarray]]:
    real_markers: list[tuple[int, int, int, np.ndarray]] = []
    for marker_color in MARKER_COLORS:
        glyph_color, mask = _glyph_for_scale(marker_color, factor)
        coords = np.argwhere(g == marker_color)
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if _is_legend_label(g, rr, cc, marker_color, factor):
                continue
            real_markers.append((rr, cc, glyph_color, mask))
    return real_markers


def _render_expanded_output(
    input_shape: tuple[int, int],
    real_markers: list[tuple[int, int, int, np.ndarray]],
) -> np.ndarray:
    h, w = input_shape
    out_h = h
    out_w = w
    for r, c, _glyph_color, mask in real_markers:
        out_h = max(out_h, r + mask.shape[0])
        out_w = max(out_w, c + mask.shape[1])

    out = np.zeros((out_h, out_w), dtype=int)
    for r, c, glyph_color, mask in real_markers:
        _paint(out, r, c, glyph_color, mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored control pixel into a uniformly enlarged glyph and grow the canvas to fit.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - At least one legend sample is present: a fully drawn enlarged glyph in its output color, plus a same-type marker one full glyph-size below-right of that sample.

    Rule:
    - Infer one shared enlargement factor from the legend, scale the original 4x4 glyph dictionary by pixel replication, and replace each real marker by its enlarged glyph in the usual mapped output color.
    - Treat marker pixels that sit exactly one enlarged glyph-size below-right of a matching legend sample as labels, so they do not generate output glyphs.
    - Do not crop enlarged glyphs at the input border; instead, expand the output grid downward and rightward until every real glyph fits completely.

    Key insight:
    - The legend no longer only teaches the shape-color mapping; its bounding-box size also reveals the global magnification factor, and that same size determines both label detection and how much the output grid must grow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *MARKER_COLORS, *OUTPUT_COLORS})

    factor = _detect_scale(g)
    real_markers = _collect_real_markers(g, factor)
    assert real_markers
    return _render_expanded_output(g.shape, real_markers)


def _build_real_marker_grid(scale: int) -> tuple[np.ndarray, list[tuple[int, int, int]], int]:
    size = 4 * scale
    rows = random.randint(2, 3 if scale == 2 else 2)
    cols = random.randint(2, 3 if scale == 2 else 2)
    spacing_r = size + random.randint(1, 2)
    spacing_c = size + random.randint(1, 2)

    min_h = size + 3 + (rows - 1) * spacing_r
    min_w = size + 3 + (cols - 1) * spacing_c
    height = random.randint(min_h, min_h + 3)
    width = random.randint(min_w, min_w + 3)

    base_r = height - 1 - (rows - 1) * spacing_r
    base_c = width - 1 - (cols - 1) * spacing_c
    anchors = [(base_r + dr * spacing_r, base_c + dc * spacing_c) for dr in range(rows) for dc in range(cols)]

    cell_count = len(anchors)
    max_used = 3 if cell_count >= 6 else 2
    used_count = random.randint(2, max_used)
    used_colors = random.sample(list(MARKER_COLORS), used_count)

    assignments = []
    for color in used_colors:
        assignments.extend([color, color])
    while len(assignments) < cell_count:
        assignments.append(random.choice(used_colors))
    random.shuffle(assignments)

    markers = [(r, c, color) for (r, c), color in zip(anchors, assignments, strict=True)]
    grid = np.zeros((height, width), dtype=int)
    for r, c, color in markers:
        grid[r, c] = color

    return grid, markers, random.choice(used_colors)


def _add_legend(grid: np.ndarray, marker_color: int, scale: int) -> None:
    glyph_color, mask = _glyph_for_scale(marker_color, scale)
    gh, gw = mask.shape
    assert gh == gw
    grid[:gh, :gw][mask] = glyph_color
    grid[gh, gw] = marker_color


def _count_edge_triggers(markers: list[tuple[int, int, int]], input_shape: tuple[int, int], scale: int) -> int:
    h, w = input_shape
    size = 4 * scale
    return sum(int(r + size > h or c + size > w) for r, c, _color in markers)


def generate_input() -> np.ndarray:
    """
    Build a clean legend-plus-marker-lattice input that forces enlarged glyphs to extend past the original border.
    """
    for _ in range(GEN_RETRIES):
        scale = random.choice(SCALE_FACTORS)
        grid, markers, legend_color = _build_real_marker_grid(scale)
        _add_legend(grid, legend_color, scale)

        edge_trigger_count = _count_edge_triggers(markers, grid.shape, scale)
        if edge_trigger_count <= 0:
            continue

        expected = _render_expanded_output(
            grid.shape,
            [(r, c, _glyph_for_scale(color, scale)[0], _glyph_for_scale(color, scale)[1]) for r, c, color in markers],
        )
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        if actual.shape == grid.shape:
            continue
        return grid

    raise RuntimeError("failed to generate a valid enlarged-and-expanded dfadab01 instance")
