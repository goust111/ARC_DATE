# Combo variant of dfadab01 (index 4)
# combo_spec: D2/1+D4/1
# Name: 局部动态映射+完整图例

from __future__ import annotations

import random

import numpy as np


MARKER_COLORS: tuple[int, ...] = (2, 3, 5, 8)
BASE_GLYPH_COLOR_BY_MARKER: dict[int, int] = {2: 4, 3: 1, 5: 6, 8: 7}
OUTPUT_COLORS: tuple[int, ...] = tuple(BASE_GLYPH_COLOR_BY_MARKER[c] for c in MARKER_COLORS)
GRID_SIZE = 20
GLYPH_SIZE = 4
MASK_AREA = 8
LEGEND_COL = 0
LEGEND_LABEL_COL = 4
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
MAX_GEN_ATTEMPTS = 400

BASE_MASK_BY_MARKER: dict[int, np.ndarray] = {
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
ALLOWED_COLORS = {0, *MARKER_COLORS, *OUTPUT_COLORS}


# Design analysis
# ===============
# concretization_and_trace:
#   - 融合后输入固定是 20x20。左侧 5 列是完整图例区：4 个 4x4 样本纵向堆叠，
#     top-left 行分别为 0/5/10/15；每个样本右下角偏移 (+4,+4) 有一个单像素
#     label，label 颜色属于 {2,3,5,8}，表示“这种 marker 该展开成左上那块样本”。
#   - 与原题不同，样本的 4x4 mask 不再来自全局常量库，而是每题重新随机生成；
#     同时 marker 色 -> 输出色也不再固定，而是通过图例局部解码。
#   - 具体追踪：若本题第一个图例是“橙色随机 4x4 样本，label=2”，另一个图例是
#     “蓝色随机 4x4 样本，label=8”，则所有真实的 `2` 都要展开成那块橙色随机
#     样本，所有真实的 `8` 都要展开成那块蓝色随机样本。原题会把 `2` 画成固定黄框，
#     融合后则必须先读完整图例，拿到 marker -> (输出色, mask) 的局部字典。
# complexity_calibration:
#   - 每题都展示完整 4 项图例，并强制真实区域使用全部 4 种 marker，且每种至少
#     重复 2 次，避免“读到半个图例也能猜完”。
#   - 随机 mask 限制为 4x4 内恰好 8 个像素、8 邻接连通、每行每列至少 1 个像素，
#     这样图案足够像字形而不是噪声。
#   - 四个随机 mask 两两汉明距离至少 4，并且都与原题四个固定 mask 保持距离，
#     降低视觉混淆，也避免退化回原题字典。
#   - 颜色映射使用 4 色置换并强制至少 2 个 marker 偏离原题固定映射，使 D2 的
#     动态颜色映射在样本里稳定出现。
# hidden_assumption_review:
#   - 原题 `GLYPHS` 直接把 marker 颜色绑定到输出色和固定 mask；融合后这个前提
#     完全失效，solver 不能再硬编码任何 marker -> glyph 关系。
#   - 原 `_has_source_glyph(... color, mask)` 依赖已知固定 mask 才能识别 legend label；
#     现在必须改成从 label 左上 4x4 patch 直接读取颜色与 mask。
#   - 原主循环边扫描边跳过图例 label，默认“知道该 label 对应哪个 glyph”；融合后
#     必须先完整解码 4 个 legend entry，再统一处理真实 marker。
#   - 新 solver 隐含假设：每种 marker 色恰有 1 个合法 legend label，且该 label 左上
#     4x4 patch 只含一种输出色并满足随机字形约束。生成器需要构造性保证这些条件。
# placement_legality:
#   - 图例固定占据列 0..4；真实 marker 只放在列 >= 8 的规则格点，因此真实 glyph
#     展开后不会碰到图例区。
#   - 真实 marker 行列间距均 >= 4，保证输出中的 4x4 glyph 彼此不重叠；选择 17/18
#     作为末行/末列时会自然产生边界裁剪，保留原题这一现象。
# visual_observability:
#   - 解题者先看到左侧完整字典，再看到右侧待展开 marker 场景；图例和主体分区清楚。
#   - legend 顺序随机、颜色映射随机、mask 也随机，因此不能靠“第几行是什么”或旧题
#     固定字典猜测，只能逐项解码。
# tidiness_and_inferability:
#   - 图例与主体无叠压，输出只保留右侧展开结果，画面仍是整洁的 4x4 模块拼接。
#   - 颜色种类仍只用原题的 4 个输出色；颜色变化总是伴随一个完整 4x4 图例样本，
#     人类可以从结构线索而不是数字 id 推断映射。
# fusion_metrics_used_in_generation:
#   - `color_effect_cells`: 若忽略 D2、仍使用原题固定输出色，但保留当前随机 mask，
#     与真实输出相比有多少像素不同；要求 > 0，确保局部颜色映射确实生效。
#   - `mask_effect_cells`: 若忽略 D4、仍使用原题固定 mask，但保留当前颜色映射，
#     与真实输出相比有多少像素不同；要求 > 0，确保随机字形确实生效。


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


def _mask_connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
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


def _is_valid_glyph_mask(mask: np.ndarray) -> bool:
    return bool(
        mask.shape == (GLYPH_SIZE, GLYPH_SIZE)
        and int(mask.sum()) == MASK_AREA
        and _mask_connected(mask)
        and np.all(mask.sum(axis=0) >= 1)
        and np.all(mask.sum(axis=1) >= 1)
    )


def _read_legend_entry(g: np.ndarray, r: int, c: int) -> tuple[int, np.ndarray] | None:
    if r < GLYPH_SIZE or c < GLYPH_SIZE:
        return None
    patch = g[r - GLYPH_SIZE : r, c - GLYPH_SIZE : c]
    nonzero = np.unique(patch[patch != 0])
    if nonzero.size != 1:
        return None
    glyph_color = int(nonzero[0])
    if glyph_color not in OUTPUT_COLORS:
        return None
    mask = patch == glyph_color
    if not _is_valid_glyph_mask(mask):
        return None
    return glyph_color, mask


def _decode_full_legend(
    g: np.ndarray,
) -> tuple[dict[int, tuple[int, np.ndarray]], set[tuple[int, int]]]:
    mapping: dict[int, tuple[int, np.ndarray]] = {}
    legend_labels: set[tuple[int, int]] = set()

    for marker_color in MARKER_COLORS:
        matches: list[tuple[int, int, int, np.ndarray]] = []
        for r, c in np.argwhere(g == marker_color):
            entry = _read_legend_entry(g, int(r), int(c))
            if entry is None:
                continue
            glyph_color, mask = entry
            matches.append((int(r), int(c), glyph_color, mask))

        assert len(matches) == 1, f"expected exactly one legend for marker color {marker_color}, got {len(matches)}"
        r, c, glyph_color, mask = matches[0]
        mapping[marker_color] = (glyph_color, mask)
        legend_labels.add((r, c))

    used_output_colors = {mapping[marker_color][0] for marker_color in MARKER_COLORS}
    assert used_output_colors == set(OUTPUT_COLORS), "legend must expose all four output colors"
    return mapping, legend_labels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the full per-puzzle legend, then expand each marker with its locally defined colored 4x4 glyph.

    Input structure:
    - Left side: four complete legend entries stacked vertically; each entry is a colored 4x4 sample plus a marker-color label one cell down-right.
    - Right side: isolated single-pixel markers in colors 2, 3, 5, and 8 placed on a regular lattice.
    - The legend covers all four marker colors, but its order, sample colors, and 4x4 masks can change from puzzle to puzzle.

    Rule:
    - For each label pixel, read the 4x4 patch immediately above-left of it to learn both the output color and the glyph mask owned by that marker color.
    - Replace every non-legend marker of that color with the learned 4x4 sample, anchored at the marker position and clipped by the grid boundary.
    - Remove the entire legend region and do not expand the label pixels themselves.

    Key insight:
    - There is no global marker dictionary anymore: both the color assignment and the glyph shapes are specified locally by the full legend, so solving requires decoding all legend entries before painting the scene.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape == (GRID_SIZE, GRID_SIZE)
    assert set(map(int, np.unique(g))).issubset(ALLOWED_COLORS)

    mapping, legend_labels = _decode_full_legend(g)

    out = np.zeros_like(g)
    seen_real_marker = False
    for marker_color in MARKER_COLORS:
        glyph_color, mask = mapping[marker_color]
        for r, c in np.argwhere(g == marker_color):
            cell = (int(r), int(c))
            if cell in legend_labels:
                continue
            seen_real_marker = True
            _paint_clipped(out, cell[0], cell[1], glyph_color, mask)

    assert seen_real_marker, "expected at least one real marker outside the legend"
    return out


def _random_mask() -> np.ndarray:
    while True:
        picks = random.sample(range(GLYPH_SIZE * GLYPH_SIZE), MASK_AREA)
        mask = np.zeros((GLYPH_SIZE, GLYPH_SIZE), dtype=bool)
        for idx in picks:
            mask[idx // GLYPH_SIZE, idx % GLYPH_SIZE] = True
        if not _is_valid_glyph_mask(mask):
            continue
        if any(int(np.sum(mask != base_mask)) < 4 for base_mask in BASE_MASK_BY_MARKER.values()):
            continue
        return mask


def _sample_masks() -> dict[int, np.ndarray]:
    chosen: list[np.ndarray] = []
    mapping: dict[int, np.ndarray] = {}
    for marker_color in MARKER_COLORS:
        while True:
            mask = _random_mask()
            if any(int(np.sum(mask != other)) < 4 for other in chosen):
                continue
            mapping[marker_color] = mask
            chosen.append(mask)
            break
    return mapping


def _sample_color_mapping() -> dict[int, int]:
    while True:
        shuffled = random.sample(list(OUTPUT_COLORS), len(OUTPUT_COLORS))
        mapping = {marker_color: shuffled[i] for i, marker_color in enumerate(MARKER_COLORS)}
        moved_count = sum(int(mapping[marker_color] != BASE_GLYPH_COLOR_BY_MARKER[marker_color]) for marker_color in MARKER_COLORS)
        if moved_count >= 2:
            return mapping


def _body_positions() -> list[tuple[int, int]]:
    rows = list(random.choice(BODY_ROW_LAYOUTS))
    cols = list(random.choice(BODY_COL_LAYOUTS))
    positions = [(int(r), int(c)) for r in rows for c in cols]
    random.shuffle(positions)
    keep = random.randint(max(8, int(np.ceil(0.6 * len(positions)))), len(positions))
    return positions[:keep]


def _sample_real_markers() -> list[tuple[int, int, int]]:
    positions = _body_positions()
    colors = [marker_color for marker_color in MARKER_COLORS for _ in range(2)]
    colors.extend(random.choices(MARKER_COLORS, k=len(positions) - len(colors)))
    random.shuffle(colors)
    return [(r, c, colors[i]) for i, (r, c) in enumerate(positions)]


def _build_output(
    color_by_marker: dict[int, int],
    mask_by_marker: dict[int, np.ndarray],
    real_markers: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r, c, marker_color in real_markers:
        _paint_clipped(out, int(r), int(c), int(color_by_marker[marker_color]), mask_by_marker[marker_color])
    return out


def _build_input(
    color_by_marker: dict[int, int],
    mask_by_marker: dict[int, np.ndarray],
    real_markers: list[tuple[int, int, int]],
) -> np.ndarray:
    inp = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legend_order = list(MARKER_COLORS)
    random.shuffle(legend_order)
    for row_start, marker_color in zip(LEGEND_ROW_STARTS, legend_order):
        _paint_clipped(inp, int(row_start), LEGEND_COL, int(color_by_marker[marker_color]), mask_by_marker[marker_color])
        inp[int(row_start) + GLYPH_SIZE, LEGEND_LABEL_COL] = marker_color

    for r, c, marker_color in real_markers:
        inp[int(r), int(c)] = marker_color
    return inp


def _count_diff(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _build_case() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    color_by_marker = _sample_color_mapping()
    mask_by_marker = _sample_masks()
    real_markers = _sample_real_markers()

    expected = _build_output(color_by_marker, mask_by_marker, real_markers)
    inp = _build_input(color_by_marker, mask_by_marker, real_markers)

    color_fixed_output = _build_output(BASE_GLYPH_COLOR_BY_MARKER, mask_by_marker, real_markers)
    mask_fixed_output = _build_output(color_by_marker, BASE_MASK_BY_MARKER, real_markers)
    original_output = _build_output(BASE_GLYPH_COLOR_BY_MARKER, BASE_MASK_BY_MARKER, real_markers)

    moved_count = sum(int(color_by_marker[marker_color] != BASE_GLYPH_COLOR_BY_MARKER[marker_color]) for marker_color in MARKER_COLORS)
    min_pairwise_mask_distance = min(
        int(np.sum(mask_by_marker[a] != mask_by_marker[b]))
        for i, a in enumerate(MARKER_COLORS)
        for b in MARKER_COLORS[i + 1 :]
    )
    min_base_distance = min(
        int(np.sum(mask_by_marker[marker_color] != base_mask))
        for marker_color in MARKER_COLORS
        for base_mask in BASE_MASK_BY_MARKER.values()
    )
    color_effect_cells = _count_diff(expected, color_fixed_output)
    mask_effect_cells = _count_diff(expected, mask_fixed_output)
    baseline_diff_cells = _count_diff(expected, original_output)

    assert moved_count >= 2
    assert min_pairwise_mask_distance >= 4
    assert min_base_distance >= 4
    assert color_effect_cells > 0
    assert mask_effect_cells > 0
    assert baseline_diff_cells > 0

    solved = main(inp)
    assert np.array_equal(solved, expected)

    meta = {
        "moved_count": moved_count,
        "min_pairwise_mask_distance": min_pairwise_mask_distance,
        "min_base_distance": min_base_distance,
        "color_effect_cells": color_effect_cells,
        "mask_effect_cells": mask_effect_cells,
        "baseline_diff_cells": baseline_diff_cells,
        "real_marker_count": len(real_markers),
    }
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            inp, _, _ = _build_case()
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")


def generate_pair() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            inp, out, _ = _build_case()
            return inp, out
        except Exception:
            continue
    raise RuntimeError(f"generate_pair: failed after {MAX_GEN_ATTEMPTS} attempts")
