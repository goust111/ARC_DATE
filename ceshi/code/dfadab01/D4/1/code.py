# Variant of dfadab01
# Dimension: D4 — 样本字形展示与消失 (symbol_instructions)
# Change: 输入中可能包含一个样本字形（完整4x4图案）及其右下角一格处的单像素标签（标签颜色=该字形对应的标记色），样本和标签在输... → 多个样本字形作为完整图例

from __future__ import annotations

import random

import numpy as np


MARKER_TO_GLYPH_COLOR: dict[int, int] = {2: 4, 3: 1, 5: 6, 8: 7}
MARKER_COLORS: tuple[int, ...] = tuple(MARKER_TO_GLYPH_COLOR)
GLYPH_COLORS: tuple[int, ...] = tuple(MARKER_TO_GLYPH_COLOR[c] for c in MARKER_COLORS)
GRID_SIZE = 20
GLYPH_SIZE = 4
LEGEND_COL = 0
LEGEND_ROW_STARTS = (0, 5, 10, 15)
BODY_ROW_LAYOUTS = (
    (0, 4, 8, 12, 16),
    (1, 5, 9, 13, 17),
    (2, 6, 10, 14, 18),
    (0, 5, 10, 15),
    (2, 7, 12, 17),
)
BODY_COL_LAYOUTS = (
    (8, 12, 16),
    (9, 13, 17),
    (10, 14, 18),
    (8, 13, 18),
)
MASK_AREA = 8
MAX_GEN_ATTEMPTS = 400

# Design analysis
# -
# concretization_and_trace:
#   - 输入左侧是完整图例区：4 个 4x4 样本纵向堆叠，每个样本右下角偏移 (+4,+4) 放 1 个单像素标签。
#   - 标签仍用 marker 色 {2,3,5,8}；样本仍用对应输出色 {4,1,6,7}，这样“颜色变换”部分保持原题不变，变化只落在字形来源。
#   - 主体区仍是单像素 marker 规则网格；输出仍把每个真实 marker 扩展成 4x4 图案，并在边界裁剪。
#   - 具体追踪：原题里 color=2 时，solver 直接使用硬编码 frame mask，并把图例 label 当作特殊跳过点。
#     现在 color=2 时，solver 必须先在输入里找到一个“黄色 4x4 样本 + 右下角红色标签”的 legend entry，从该 4x4 patch 学出 mask，
#     再用这个 learned mask 去扩展所有真实的红色 marker。差异在于“mask 来源”从固定常量改成输入中的完整图例。
# complexity_calibration:
#   - 每题都展示全部 4 类 legend entry，且主体区强制出现全部 4 类 marker；这样解题者必须读完整图例，不能只靠一两个样本。
#   - 每个样本 mask 随机生成，但限制为 4x4 内恰好 8 个像素、8 邻接连通、每行每列至少 1 个像素，避免退化成噪声点阵。
#   - 4 个 mask 之间要求有足够的汉明距离，降低视觉近似和歧义。
# implicit_assumption_review:
#   - 原 main() 的 `for marker_color, (glyph_color, mask) in GLYPHS.items()` 假设 mask 事先固定可知；变体后这个前提失效，必须先解析 legend。
#   - 原 `_has_source_glyph()` 用固定 mask 检测“这个 marker 是否是图例标签”；变体后不能再拿固定 mask 比对，只能从标签左上角整块 4x4 patch 读取。
#   - 原 assert 允许出现少量输出色只是因为最多一个 legend；现在输入中一定有 4 个 legend，因此 solver 需要允许全部输出色并假设每种恰有 1 个 legend entry。
# placement_legality:
#   - 图例区固定在左 5 列，主体区的 marker top-left 固定在第 8 列以后；因此主体 glyph 展开后不会碰到图例。
#   - 主体区行列间距始终 >= 4，保证输出中的 4x4 glyph 彼此不重叠；末行/末列布局有时放到 17/18，从而保留原题的裁剪现象。
# observability_and_tidy_layout:
#   - 图例独立成一列，主体 marker 独立成规则网格，输入视觉上分区清晰；输出时左侧图例完全消失，变化可直接观察。
#   - legend 顺序打乱，mask 也随机，因此不能依赖“第几行对应哪种字形”或旧题固定字典。
# color_inference:
#   - 不新增颜色映射自由度，只保留原题固定的 marker 色 -> 输出色；真正变化的是 4x4 形状本身，线索完全来自可见 legend。
# new_solver_assumptions:
#   - `main()` 现在假设每个 marker 色都存在且只存在 1 个合法 legend label，其左上 4x4 patch 只含背景或对应输出色。
#   - `generate_input()` 通过固定 legend 区、固定 label offset、唯一颜色区分和 oracle 校验来保证这些假设成立。


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


def _read_legend_mask(g: np.ndarray, r: int, c: int, glyph_color: int) -> np.ndarray | None:
    if r < GLYPH_SIZE or c < GLYPH_SIZE:
        return None
    patch = g[r - GLYPH_SIZE : r, c - GLYPH_SIZE : c]
    nonzero = patch[patch != 0]
    if nonzero.size == 0:
        return None
    if not np.all(nonzero == glyph_color):
        return None
    mask = patch == glyph_color
    if not mask.any():
        return None
    return mask


def _find_legend_masks(g: np.ndarray) -> tuple[dict[int, np.ndarray], set[tuple[int, int]]]:
    learned_masks: dict[int, np.ndarray] = {}
    legend_labels: set[tuple[int, int]] = set()
    for marker_color, glyph_color in MARKER_TO_GLYPH_COLOR.items():
        for r, c in np.argwhere(g == marker_color):
            mask = _read_legend_mask(g, int(r), int(c), glyph_color)
            if mask is None:
                continue
            assert marker_color not in learned_masks, f"duplicate legend for marker {marker_color}"
            learned_masks[marker_color] = mask
            legend_labels.add((int(r), int(c)))
        assert marker_color in learned_masks, f"missing legend for marker {marker_color}"
    return learned_masks, legend_labels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the per-input legend, then expand each control pixel into its learned 4x4 glyph.

    Input structure:
    - Left side: a full legend made of four complete 4x4 sample glyphs in colors 4, 1, 6, and 7.
    - Each sample has a matching marker-color label pixel one cell below-right of its 4x4 box.
    - Elsewhere: a regular field of single-pixel markers in colors 2, 3, 5, and 8.

    Rule:
    - For each marker color, read the 4x4 sample just above-left of its legend label to learn that marker's glyph mask.
    - Replace every non-legend marker of that color with the learned 4x4 mask, painted in the same fixed output color as the legend sample.
    - Keep the glyph top-left anchored at the marker pixel and clip any part that falls outside the grid.
    - Remove the entire legend region and its label pixels from the output.

    Key insight:
    - The shape dictionary is no longer fixed across puzzles; the only way to know each marker's 4x4 pattern is to read the complete legend embedded in the input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape == (GRID_SIZE, GRID_SIZE)
    assert set(np.unique(g)).issubset({0, *MARKER_COLORS, *GLYPH_COLORS})

    learned_masks, legend_labels = _find_legend_masks(g)

    out = np.zeros_like(g)
    seen_real_marker = False
    for marker_color, glyph_color in MARKER_TO_GLYPH_COLOR.items():
        mask = learned_masks[marker_color]
        for r, c in np.argwhere(g == marker_color):
            rc = (int(r), int(c))
            if rc in legend_labels:
                continue
            seen_real_marker = True
            _paint_clipped(out, int(r), int(c), glyph_color, mask)

    assert seen_real_marker
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    stack = [cells[0]]
    seen = {cells[0]}
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < GLYPH_SIZE and 0 <= cc < GLYPH_SIZE and bool(mask[rr, cc]) and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    stack.append((rr, cc))
    return len(seen) == len(cells)


def _random_mask() -> np.ndarray:
    while True:
        picks = random.sample(range(GLYPH_SIZE * GLYPH_SIZE), MASK_AREA)
        mask = np.zeros((GLYPH_SIZE, GLYPH_SIZE), dtype=bool)
        for idx in picks:
            mask[idx // GLYPH_SIZE, idx % GLYPH_SIZE] = True
        if not _mask_connected(mask):
            continue
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        return mask


def _sample_masks() -> dict[int, np.ndarray]:
    chosen: list[np.ndarray] = []
    for marker_color in MARKER_COLORS:
        while True:
            mask = _random_mask()
            if any(int(np.sum(mask != other)) < 4 for other in chosen):
                continue
            chosen.append(mask)
            break
    return {marker_color: chosen[i] for i, marker_color in enumerate(MARKER_COLORS)}


def _body_positions() -> list[tuple[int, int]]:
    rows = list(random.choice(BODY_ROW_LAYOUTS))
    cols = list(random.choice(BODY_COL_LAYOUTS))
    positions = [(int(r), int(c)) for r in rows for c in cols]
    random.shuffle(positions)
    keep = random.randint(max(8, int(np.ceil(0.6 * len(positions)))), len(positions))
    return positions[:keep]


def _build_output(marker_masks: dict[int, np.ndarray], real_markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r, c, marker_color in real_markers:
        _paint_clipped(out, int(r), int(c), MARKER_TO_GLYPH_COLOR[marker_color], marker_masks[marker_color])
    return out


def _build_input(marker_masks: dict[int, np.ndarray], real_markers: list[tuple[int, int, int]]) -> np.ndarray:
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legend_order = list(MARKER_COLORS)
    random.shuffle(legend_order)
    for r0, marker_color in zip(LEGEND_ROW_STARTS, legend_order):
        glyph_color = MARKER_TO_GLYPH_COLOR[marker_color]
        mask = marker_masks[marker_color]
        for dr in range(GLYPH_SIZE):
            for dc in range(GLYPH_SIZE):
                if bool(mask[dr, dc]):
                    g[r0 + dr, LEGEND_COL + dc] = glyph_color
        g[r0 + GLYPH_SIZE, LEGEND_COL + GLYPH_SIZE] = marker_color

    for r, c, marker_color in real_markers:
        g[r, c] = marker_color
    return g


def _sample_real_markers() -> list[tuple[int, int, int]]:
    positions = _body_positions()
    colors = list(MARKER_COLORS)
    random.shuffle(colors)
    chosen_colors = colors[:]
    while len(chosen_colors) < len(positions):
        chosen_colors.append(random.choice(MARKER_COLORS))
    random.shuffle(chosen_colors)
    return [(r, c, chosen_colors[i]) for i, (r, c) in enumerate(positions)]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        marker_masks = _sample_masks()
        real_markers = _sample_real_markers()
        expected = _build_output(marker_masks, real_markers)
        inp = _build_input(marker_masks, real_markers)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
