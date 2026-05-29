# Combo variant of dfadab01 (index 10)
# combo_spec: D4/1+D11/0
# Name: 完整图例+全字形样本展示

from __future__ import annotations

import random

import numpy as np


MARKER_TO_GLYPH_COLOR: dict[int, int] = {2: 4, 3: 1, 5: 6, 8: 7}
MARKER_COLORS: tuple[int, ...] = tuple(MARKER_TO_GLYPH_COLOR)
GLYPH_COLORS: tuple[int, ...] = tuple(MARKER_TO_GLYPH_COLOR[c] for c in MARKER_COLORS)
GRID_SIZE = 20
GLYPH_SIZE = 4
LEGEND_COL = 0
LEGEND_LABEL_COL = LEGEND_COL + GLYPH_SIZE
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
MIN_MASK_DISTANCE = 4
MAX_GEN_ATTEMPTS = 400

# Design analysis
# -
# concretization_and_trace:
#   - 融合后输入被分成两个清晰区域：左侧是完整图例，右侧是主体 marker 网格。左侧固定出现 4 个 4x4 样本，每个样本右下角偏移
#     (+4,+4) 放 1 个单像素标签，4 个标签颜色正好覆盖 {2,3,5,8}；右侧仍是若干真实单像素 marker。
#   - 具体追踪一个 marker=2 的例子：原题 solver 看到红色像素后，直接调用硬编码的“黄色 frame” mask，并且只在像素左上角恰好存在
#     那个固定 frame 时才把它识别成 legend label。融合后 solver 先在整张图里找到一个“黄色 4x4 样本 + 其右下角红色标签”，从该样本
#     直接读出本题的 4x4 mask；随后所有其他红色真实 marker 都展开成这个刚读出的形状。差异在于 shape 来源从固定常量变成输入图例。
# complexity_calibration:
#   - D4/1 与 D11/0 叠加后的关键不是“可能出现多个样本”，而是“每题必须出现全部四种样本”。因此 generator 强制四种标签全出现，且
#     主体区也强制四种真实 marker 至少各出现一次，避免任务退化成只需读单个样本或只用到少数字形。
#   - 每个 4x4 样本限制为恰好 8 个着色格、8 邻接连通、每行每列至少有 1 个着色格，并要求任意两种样本之间至少有 4 格汉明距离，
#     这样视觉上像“字形”而不是噪声，也减少不同样本过于相似导致的歧义。
# implicit_assumption_review:
#   - 原 `GLYPHS` 常量及 `for marker_color, (glyph_color, mask) in GLYPHS.items()` 依赖“字形字典跨题固定”；融合后这个前提失效，
#     所以先改成从输入里学习 `learned_masks`，再进行扩展。
#   - 原 `_has_source_glyph()` 依赖“知道样本的固定 mask 才能判定某个 marker 是 legend label”；融合后不能先验知道 mask，
#     因而改成更弱但足够的判定：标签左上 4x4 patch 只允许出现该输出色和背景，并且非空。
#   - 原代码默认输入中输出色只是可选的少量 legend 噪声；融合后四种输出色都会系统性出现于左侧 legend，因此输入颜色断言和解析顺序
#     都必须兼容完整图例。
# placement_legality:
#   - 左侧 legend 占用列 0..4，其中样本画在列 0..3、标签落在列 4；主体 marker 的 top-left 只放在列 8 及以后，所以主体 glyph 展开
#     到 4x4 后也不会碰到 legend。
#   - 主体候选位置的行列间距都至少为 4，因此输出里的 4x4 glyph 彼此不会重叠；最后一行/列允许取 17 或 18，保留原题“靠边裁剪”的现象。
# visual_observability_and_tidy_layout:
#   - 解题者能直接看到“左边是四条样本-标签对照表，右边是待解码 marker 网格”，输入分区明确，输出中左侧整块消失，只剩右侧扩张后的图案，
#     变化非常可见。
#   - legend 顺序被打乱，样本形状也按题随机，因此不能凭固定行序或旧题字典作答，必须真正读取完整图例。
# color_inference_and_rule_inferability:
#   - 颜色映射仍沿用原题固定关系 2->4, 3->1, 5->6, 8->7；新的自由度只落在 4x4 形状。这样人类只需推断“标签色对应哪个样本形状”，
#     而不必同时猜颜色重映射和形状重映射，保证 2-4 个样例内可归纳。
# new_solver_assumptions_and_generator_guards:
#   - `main()` 现在假设每种 marker 色都恰有一个合法 legend label，且它左上 4x4 区域非空、只含对应 glyph 色与背景。
#   - `main()` 还假设真实 marker 不会落在 legend 区，输入中的 glyph 色不会出现在主体区；generator 通过固定分区、唯一标签偏移和 oracle
#     校验来保证这些前提成立。


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
    if nonzero.size == 0 or not np.all(nonzero == glyph_color):
        return None
    return patch == glyph_color


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
    Read the full per-input legend, then expand each real marker into its learned 4x4 glyph.

    Input structure:
    - The left side contains a complete legend: all four glyph types appear as 4x4 samples in colors 4, 1, 6, and 7.
    - Each legend sample has a matching marker-color label pixel placed one cell below-right of the sample box.
    - The rest of the grid contains the actual single-pixel markers in colors 2, 3, 5, and 8.

    Rule:
    - For each marker color, locate its legend label and read the 4x4 sample immediately above-left of that label.
    - Use that learned 4x4 mask to replace every non-legend marker of the same color, painting with the sample's fixed glyph color.
    - Keep each expansion anchored at the marker's top-left cell, clipping any part that would leave the grid.
    - Remove the full legend region and all label pixels from the output.

    Key insight:
    - The 4x4 alphabet is no longer fixed across puzzles; the only reliable dictionary is the complete legend embedded in the input itself.
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
    ordered_masks: list[np.ndarray] = []
    for _marker_color in MARKER_COLORS:
        while True:
            mask = _random_mask()
            if any(int(np.sum(mask != other)) < MIN_MASK_DISTANCE for other in chosen):
                continue
            chosen.append(mask)
            ordered_masks.append(mask)
            break
    return {marker_color: ordered_masks[i] for i, marker_color in enumerate(MARKER_COLORS)}


def _body_positions() -> list[tuple[int, int]]:
    rows = list(random.choice(BODY_ROW_LAYOUTS))
    cols = list(random.choice(BODY_COL_LAYOUTS))
    positions = [(int(r), int(c)) for r in rows for c in cols]
    random.shuffle(positions)
    min_keep = max(len(MARKER_COLORS), int(np.ceil(0.6 * len(positions))))
    keep = random.randint(min_keep, len(positions))
    return positions[:keep]


def _sample_real_markers() -> list[tuple[int, int, int]]:
    positions = _body_positions()
    forced_colors = list(MARKER_COLORS)
    random.shuffle(forced_colors)
    while len(forced_colors) < len(positions):
        forced_colors.append(random.choice(MARKER_COLORS))
    random.shuffle(forced_colors)
    return [(r, c, forced_colors[i]) for i, (r, c) in enumerate(positions)]


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
        g[r0 + GLYPH_SIZE, LEGEND_LABEL_COL] = marker_color

    for r, c, marker_color in real_markers:
        g[r, c] = marker_color
    return g


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
