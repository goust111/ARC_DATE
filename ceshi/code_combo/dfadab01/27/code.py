# Combo variant of dfadab01 (index 27)
# combo_spec: D8/1+D9/2
# Name: 邻近色决定字形色+迭代放置
"""
design_analysis:
  concretization:
    preserved:
      - 输入仍是黑底上的单像素标记，标记色继续只从 {2,3,5,8} 中取值。
      - 标记自己的颜色仍只负责选择哪一种固定 4x4 字形轮廓；左上角对齐与边界裁剪也保持原题做法。
      - 为兼容原题遗留形式，solver 仍会跳过“4x4 示例字形 + 右下标签像素”的旧式 legend 标签。
    changed:
      - D8 部分具体化为“最近邻标记的颜色直接作为字形填充色”，不再经过原题 2->4、3->1、5->6、8->7 的固定输出映射。
      - D9 部分具体化为“只检查右下角这一处角点是否生成新标记”。这样保留了多轮传播的关键逻辑，
        同时避免四角同时分叉造成大量重叠与肉眼不可读的画面。
      - 每一轮先把上一轮产生的新标记整体收集出来，再用“当前已知的全部标记”重算最近邻；同一波新标记彼此也能互相影响传色。
    trace_example:
      original:
        - 两个相距最近的标记各自直接画成固定颜色的 4x4 字形，处理只做一轮，不会产生后续标记。
      fused:
        - 若 (5,4) 的 5 标记最近的是 (10,0) 的 2 标记，则它画出的仍是 5 的对角块轮廓，但整块改涂成颜色 2。
        - 因为 5 字形的右下角是实心，(8,7) 会出现一个新 2 标记；下一轮要把所有新标记一起拿来，再与旧标记共同重算最近邻后继续放置。
        - 若另一条链在同轮也生成了相邻的新标记，它可能改变这个新 2 标记的最近邻颜色，因此第二轮的颜色不一定等于“只看原始根标记”时的结果。
      difference:
        - 原题是“自己决定形状，自己决定颜色，一次性完成”。
        - 融合后是“自己决定形状，最近邻决定颜色，而且右下角还能把颜色继续传成下一轮的新标记”。
  complexity_calibration:
    - 为了让 D8 变化显眼，生成器固定使用 4 个根标记，并强制至少一半已放置字形发生“颜色 != 自身标记色”的重着色。
    - 为了让 D9 变化显眼，生成器只采样会产生至少 4 个后续标记的模板，并要求最终输出与“只做第一轮”的基线不同。
    - 为了让“每轮重算最近邻”不是空话，生成器进一步要求最终输出与“传播存在，但新标记只看原始根标记传色”的基线不同。
    - 颜色模式只从少量人工筛过的组合里抽样，避免自由度太大导致最近邻平局或规律太隐蔽。
  implicit_assumption_review:
    original_main:
      - 原版 `GLYPHS` 把“轮廓”和“固定输出色”绑死；融合后必须拆成单独的 `marker -> mask`，颜色则在运行时计算。
      - 原版一趟扫描就能结束；融合后这个前提失效，必须显式维护 `known markers` 与按波次推进的 `wave frontier`。
      - 原版认为输入里不会再出现会被当成新标记的输出色；融合后输出色本身就是标记色，因此需要定义明确的种子来源。
    fused_main:
      - 右下角传播的前提是新种子只来自可见、唯一的右下角；生成器因此只使用已验证的布局，避免冲突颜色种子撞在同一坐标。
      - 最近邻允许等距，但等距最近邻的颜色必须一致；生成器会拒绝出现异色平局的样本。
      - 新标记在某一轮的传色必须允许参考同轮其它新标记；solver 先整轮收集新标记，再进入下一轮统一重算最近邻。
  placement_legality:
    - 画布固定为 20x20，根标记只从两个人工校准模板及其小范围平移里选择，保证初始结构整齐且有足够的东南方向传播空间。
    - 生成器对每个候选输入都执行 output-first 验证：先模拟完整融合规则，再检查 `main(input)` 是否精确复现该输出。
    - 所有候选样本都要通过两类一致性检查：最近邻无异色平局，新旧波次生成的同坐标标记若重复则颜色必须一致。
  observability_and_cleanliness:
    - 同一输入里会出现“同形异色”的字形，足以排除原题的固定配色解释。
    - 输出里还会出现一条明显的右下阶梯式链条，足以排除“一次性替换”解释。
    - 传播只走一个角点，避免四角分叉把多个字形压在同一小块区域里；保留了迭代感，也保持了视觉整洁。
  generator_checks:
    - changed_fraction: `recolored_glyphs / total_placed_glyphs >= 1/2`
    - repeated_shape_multicolor: 至少一种字形轮廓在同一张图里出现两种不同填充色
    - propagation_diff_cells: 融合输出与“只画第一轮根标记”基线的差异像素数 > 0
    - recomputed_diff_cells: 融合输出与“允许传播但新标记只参考原始根标记”基线的差异像素数 > 0
"""

from __future__ import annotations

import random

import numpy as np


GLYPH_MASKS: dict[int, np.ndarray] = {
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

MARKER_COLORS = tuple(sorted(GLYPH_MASKS))
ORIGINAL_GLYPH_COLORS = (1, 4, 6, 7)
LEGEND_PAINT_COLORS = tuple(range(1, 9))
ALLOWED_COLORS = {0, *MARKER_COLORS, *ORIGINAL_GLYPH_COLORS}
GRID_SIZE = 20
GEN_RETRIES = 200

# Each tuple is (base anchor positions, max row shift, max col shift).
LAYOUT_SPECS: tuple[tuple[tuple[tuple[int, int], ...], int, int], ...] = (
    (((0, 0), (10, 0), (5, 4), (5, 9)), 3, 5),
    (((0, 0), (0, 10), (4, 5), (9, 5)), 5, 3),
)

# Colors were pre-filtered so that the fused metrics stay non-degenerate.
ROOT_COLOR_PATTERNS: tuple[tuple[int, int, int, int], ...] = (
    (3, 2, 5, 3),
    (3, 2, 8, 3),
    (3, 5, 2, 3),
    (3, 5, 8, 3),
    (3, 8, 2, 3),
    (3, 8, 5, 3),
)


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _is_legend_label(g: np.ndarray, r: int, c: int, marker_color: int) -> bool:
    mask = GLYPH_MASKS[marker_color]
    return any(_has_source_glyph(g, r - 4, c - 4, paint_color, mask) for paint_color in LEGEND_PAINT_COLORS)


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


def _collect_real_markers(g: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(np.isin(g, MARKER_COLORS)):
        rr = int(r)
        cc = int(c)
        marker_color = int(g[rr, cc])
        if _is_legend_label(g, rr, cc, marker_color):
            continue
        markers.append((rr, cc, marker_color))
    markers.sort()
    return markers


def _nearest_donor_color(target: tuple[int, int, int], markers: list[tuple[int, int, int]]) -> int:
    r, c, _ = target
    distances: list[tuple[int, int]] = []
    for rr, cc, color in markers:
        if (rr, cc) == (r, c):
            continue
        distances.append((abs(r - rr) + abs(c - cc), int(color)))
    assert distances, "each marker must have another marker to donate its color"
    best_dist = min(dist for dist, _ in distances)
    nearest_colors = {color for dist, color in distances if dist == best_dist}
    assert len(nearest_colors) == 1, "nearest-neighbor color must be unambiguous"
    return next(iter(nearest_colors))


def _render_from_roots(
    shape: tuple[int, int],
    roots: list[tuple[int, int, int]],
    donor_mode: str = "all_known",
    first_wave_only: bool = False,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    h, w = shape
    out = np.zeros(shape, dtype=int)
    root_reference = list(roots)
    known: dict[tuple[int, int], int] = {(r, c): color for r, c, color in roots}
    wave = list(roots)
    waves: list[list[tuple[int, int, int, int]]] = []

    while wave:
        if donor_mode == "all_known":
            reference = [(r, c, color) for (r, c), color in sorted(known.items())]
        elif donor_mode == "roots_only":
            reference = root_reference
        else:
            raise ValueError(f"unsupported donor_mode: {donor_mode}")

        placements: list[tuple[int, int, int, int]] = []
        next_wave_map: dict[tuple[int, int], int] = {}

        for r, c, marker_color in wave:
            donor_color = _nearest_donor_color((r, c, marker_color), reference)
            placements.append((r, c, marker_color, donor_color))
            _paint_clipped(out, r, c, donor_color, GLYPH_MASKS[marker_color])

            if not bool(GLYPH_MASKS[marker_color][3, 3]):
                continue
            nr, nc = r + 3, c + 3
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            prev_known = known.get((nr, nc))
            if prev_known is not None:
                assert prev_known == donor_color, "repeated spawned marker must keep the same color"
                continue
            prev_pending = next_wave_map.get((nr, nc))
            if prev_pending is None:
                next_wave_map[(nr, nc)] = donor_color
            else:
                assert prev_pending == donor_color, "same-wave spawned marker must be color-consistent"

        waves.append(placements)
        if first_wave_only:
            break

        wave = []
        for (r, c), marker_color in sorted(next_wave_map.items()):
            known[(r, c)] = marker_color
            wave.append((r, c, marker_color))

    return out, waves


def _single_wave_output(shape: tuple[int, int], roots: list[tuple[int, int, int]]) -> np.ndarray:
    out, _ = _render_from_roots(shape, roots, donor_mode="all_known", first_wave_only=True)
    return out


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    roots = _collect_real_markers(g)
    assert len(roots) >= 2

    fused_out, fused_waves = _render_from_roots(g.shape, roots, donor_mode="all_known")
    root_only_out, _ = _render_from_roots(g.shape, roots, donor_mode="roots_only")
    first_wave_out = _single_wave_output(g.shape, roots)

    total_placed = int(sum(len(wave) for wave in fused_waves))
    recolored = int(sum(marker_color != donor_color for wave in fused_waves for _, _, marker_color, donor_color in wave))
    by_shape: dict[int, set[int]] = {marker_color: set() for marker_color in MARKER_COLORS}
    counts: dict[int, int] = {marker_color: 0 for marker_color in MARKER_COLORS}
    for wave in fused_waves:
        for _, _, marker_color, donor_color in wave:
            by_shape[marker_color].add(donor_color)
            counts[marker_color] += 1

    repeated_shape_multicolor = int(
        any(counts[marker_color] >= 2 and len(by_shape[marker_color]) >= 2 for marker_color in MARKER_COLORS)
    )
    return {
        "initial_markers": int(len(roots)),
        "wave_count": int(len(fused_waves)),
        "total_placed_glyphs": total_placed,
        "generated_markers": int(total_placed - len(roots)),
        "recolored_glyphs": recolored,
        "repeated_shape_multicolor": repeated_shape_multicolor,
        "propagation_diff_cells": int(np.count_nonzero(fused_out != first_wave_out)),
        "recomputed_diff_cells": int(np.count_nonzero(fused_out != root_only_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each marker into its fixed glyph shape, tint it by the nearest other marker, and keep iterating from the lower-right corner.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Old-style legend labels may still appear as a 4x4 sample glyph plus a matching marker one cell below-right of that sample.
    - Real scene content is the set of non-legend marker pixels; output-colored sample glyphs, if any, are explanatory only.

    Rule:
    - A marker's own color chooses the 4x4 glyph mask, but the glyph's paint color is the color of the nearest other real marker by Manhattan distance.
    - Paint every marker in the current wave at its top-left anchor, clipping at the border as usual.
    - If that glyph has a filled lower-right corner, the painted cell at (+3,+3) becomes a new marker for the next wave; after collecting a whole wave of such markers, recompute nearest-neighbor colors on the enlarged marker set and continue.

    Key insight:
    - Shape selection, color transfer, and propagation are now coupled: solving requires simulating wave-by-wave seed growth while re-evaluating proximity after each wave, rather than doing a one-pass fixed-color substitution.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(int(x) for x in np.unique(g)).issubset(ALLOWED_COLORS)

    roots = _collect_real_markers(g)
    assert len(roots) >= 2, "need at least two real markers for proximity-based color transfer"
    out, _ = _render_from_roots(g.shape, roots, donor_mode="all_known")
    return out


def _sample_roots() -> list[tuple[int, int, int]]:
    base_positions, max_row_shift, max_col_shift = random.choice(LAYOUT_SPECS)
    row_shift = random.randint(0, max_row_shift)
    col_shift = random.randint(0, max_col_shift)
    colors = random.choice(ROOT_COLOR_PATTERNS)
    return [
        (r + row_shift, c + col_shift, marker_color)
        for (r, c), marker_color in zip(base_positions, colors, strict=True)
    ]


def _build_input_grid(roots: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r, c, marker_color in roots:
        grid[r, c] = marker_color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        roots = _sample_roots()
        grid = _build_input_grid(roots)

        expected, _ = _render_from_roots(grid.shape, roots, donor_mode="all_known")
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue

        metrics = fusion_metrics(grid)
        if metrics["generated_markers"] < 4:
            continue
        if metrics["wave_count"] < 3:
            continue
        if metrics["recolored_glyphs"] * 2 < metrics["total_placed_glyphs"]:
            continue
        if metrics["repeated_shape_multicolor"] <= 0:
            continue
        if metrics["propagation_diff_cells"] <= 0:
            continue
        if metrics["recomputed_diff_cells"] <= 0:
            continue
        return grid

    raise RuntimeError("failed to construct a non-degenerate D8/1+D9/2 combo input")
