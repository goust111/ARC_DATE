# Variant of dfadab01
# Dimension: D12 — 标记颜色的种类数 (meta)
# Change: 使用4种标记色（2、3、5、8），对应4种不同的字形图案... → 标记色数量逐题递增

# Design analysis
# ---------------
# design_analysis:
#   concretization_and_trace:
#     preserved_core:
#       - 单像素标记仍然触发固定 4x4 字形替换，字形字典仍是 2/3/5/8 -> 4/1/6/7。
#       - legend 仍是“4x4 输出色字形 + 其右下角一格的同类标记标签”，输出中 legend 与标签一起消失。
#       - 输出仍以黑底开始，只把真实标记扩展成字形；贴边部分继续裁剪。
#     D12_variant:
#       - 变化不在单个输入的局部几何，而在整道题的样例编排：
#         train[0] 恰好使用前 2 种标记色 {2,3}，
#         train[1] 恰好使用前 3 种标记色 {2,3,5}，
#         test     恰好使用前 4 种标记色 {2,3,5,8}。
#       - “逐题递增”因此是任务级可观察信号，而不是 main(grid) 内部的新分支。
#     small_trace:
#       - 原始流程：输入若含 {2,5} 两种标记，main 会分别把 2 扩成黄色边框，把 5 扩成粉色对角块。
#       - 变体流程：单个 case 内的替换规则不变；但 generator 会保证第一个训练例只有 {2,3}，
#         第二个训练例新增 5，测试例再新增 8。解题者从多例对照可看出“可用标记色集合按固定顺序扩张”。
#       - 差异点：原题中每例用了几种标记色是随意的；变体中这个数量本身成为稳定、可见、可归纳的元规律。
#   complexity_calibration:
#     - 只让种类数走 2 -> 3 -> 4 三档，避免过多自由度。
#     - 每个已启用颜色至少出现两次，避免“种类数”只靠单个孤立像素支撑，视觉上不稳。
#     - 新增颜色按照固定前缀顺序 {2},{3},{5},{8} 引入，而不是随机换子集；这样 2-3 个样例足够观察到“新增一种颜色”。
#   implicit_assumption_review_of_main:
#     - `assert set(np.unique(g)).issubset(...)` 依赖输入只含已知 marker/glyph 颜色；变体仍然满足，因为没有引入新颜色。
#     - `for marker_color, ... in GLYPHS.items()` 依赖合法标记类型是预先固定字典；D12 只改变“本题启用了其中几种”，不改变字典本身。
#     - `_has_source_glyph(... r-4, c-4 ...)` 依赖 legend 标签恰在样本字形右下外侧 (+4,+4)，且样本 4x4 完整在界内；generator 保证这一点。
#     - `_paint_clipped(...)` 依赖字形之间不会互相覆盖；generator 继续用 >=4 的网格间距保证。
#     - `seen_marker` 依赖每个 case 至少有一个真实标记；generator 保证每个已启用颜色至少出现两次，因此显然成立。
#   placement_legality:
#     - 标记仍放在规则网格点上，行列间距都在 4..6，保证 4x4 字形不重叠。
#     - legend 若出现，会被放进一个完整空的 5x5 区域中，既不盖住真实标记，也不制造额外歧义标签。
#     - 训练与测试例都在生成后直接用 `main(input)` 复算核对，确保 solver/generator 一致。
#   visual_observability:
#     - 人类可以直接数输入中的非黑色单像素标记颜色种类：第一例 2 种，第二例 3 种，测试例 4 种。
#     - 输出端的几何规则与原题一致，因此新增的唯一稳定差异正是“颜色种类数递增”。
#   visual_cleanliness:
#     - 标记区和 legend 区分离；legend 最多一个，且落在干净 5x5 区块。
#     - 每种颜色重复至少两次，避免一片稀疏噪声。
#   color_change_inference:
#     - 没有新增复杂颜色变换；颜色映射仍是固定字典，D12 只控制“启用多少种颜色”。
#   new_main_assumptions_after_edit:
#     - main 现在在文档字符串里明确了“任务级递增、单例局部规则不变”这一语义。
#     - generate_task() 保证任务级序列总是 [2,3,4]，因此该说明与生成数据一致。

from __future__ import annotations

import math
from typing import Any

import numpy as np


GLYPHS: dict[int, tuple[int, np.ndarray]] = {
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

MARKER_ORDER: tuple[int, ...] = (2, 3, 5, 8)
TASK_TYPE_COUNTS: tuple[int, ...] = (2, 3, 4)


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


def _render_output(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, marker_color in markers:
        glyph_color, mask = GLYPHS[marker_color]
        _paint_clipped(out, r, c, glyph_color, mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each real marker pixel into its fixed 4x4 glyph while later task examples simply enable more marker colors.

    Input structure:
    - Black background with single-pixel markers drawn from an ordered subset of {2, 3, 5, 8}; across a full task, early examples use fewer of these colors and later examples use more.
    - Some inputs also include one legend entry: a fully drawn 4x4 glyph in its output color plus a matching marker placed one cell below-right of that glyph box.

    Rule:
    - Replace every real marker by its learned 4x4 glyph: 2->yellow frame, 3->blue ring, 5->pink diagonal blocks, 8->orange X.
    - Treat the marker at (+4, +4) from a matching sample glyph as a legend label, so neither that label nor the sample glyph survives in the output.
    - Paint each glyph with its top-left corner at the marker position, clipping any pixels that fall outside the grid.

    Key insight:
    - The local transformation is still a fixed lookup table; the D12 change is task-level bookkeeping, where successive examples activate larger prefixes of the same marker alphabet.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    allowed_colors = {0, *GLYPHS.keys(), *(glyph_color for glyph_color, _mask in GLYPHS.values())}
    assert set(np.unique(g)).issubset(allowed_colors)

    out = np.zeros_like(g)
    seen_marker = False
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            if _has_source_glyph(g, int(r) - 4, int(c) - 4, glyph_color, mask):
                continue
            _paint_clipped(out, int(r), int(c), glyph_color, mask)

    assert seen_marker
    return out


def _pick_layout(size: int, num_types: int, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    candidates: list[tuple[int, int, int, int]] = []
    min_cells = max(2 * num_types, 4)
    for rows in range(2, 6):
        for cols in range(2, 6):
            if rows * cols < min_cells:
                continue
            for row_gap in range(4, 7):
                for col_gap in range(4, 7):
                    if (rows - 1) * row_gap <= size - 1 and (cols - 1) * col_gap <= size - 1:
                        candidates.append((rows, cols, row_gap, col_gap))
    if not candidates:
        raise ValueError(f"no valid layout for size={size}, num_types={num_types}")

    rows, cols, row_gap, col_gap = candidates[int(rng.integers(len(candidates)))]
    row_span = (rows - 1) * row_gap
    col_span = (cols - 1) * col_gap

    row_end_lo = max(row_span, size - 6)
    col_end_lo = max(col_span, size - 6)
    row_end = int(rng.integers(row_end_lo, size))
    col_end = int(rng.integers(col_end_lo, size))

    row_start = row_end - row_span
    col_start = col_end - col_span
    row_positions = [row_start + i * row_gap for i in range(rows)]
    col_positions = [col_start + j * col_gap for j in range(cols)]
    return row_positions, col_positions


def _choose_filled_cells(
    row_positions: list[int],
    col_positions: list[int],
    used_types: tuple[int, ...],
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    cells = [(r, c) for r in row_positions for c in col_positions]
    min_fill = max(2 * len(used_types), math.ceil(0.6 * len(cells)))
    n_fill = int(rng.integers(min_fill, len(cells) + 1))
    chosen_idx = rng.choice(len(cells), size=n_fill, replace=False)

    marker_colors = [color for color in used_types for _ in range(2)]
    extra = n_fill - len(marker_colors)
    if extra > 0:
        marker_colors.extend(int(x) for x in rng.choice(used_types, size=extra, replace=True))
    rng.shuffle(marker_colors)

    markers = [(cells[int(idx)][0], cells[int(idx)][1], marker_colors[i]) for i, idx in enumerate(chosen_idx)]
    return markers


def _place_legend(
    grid: np.ndarray,
    used_types: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[bool, tuple[int, int, int] | None]:
    if rng.random() >= 0.7:
        return False, None

    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for marker_color in used_types:
        glyph_color, mask = GLYPHS[marker_color]
        for r0 in range(h - 4):
            for c0 in range(w - 4):
                patch = grid[r0 : r0 + 5, c0 : c0 + 5]
                if np.any(patch != 0):
                    continue
                glyph_patch = grid[r0 : r0 + 4, c0 : c0 + 4]
                if np.any(glyph_patch[mask] != 0):
                    continue
                candidates.append((r0, c0, marker_color))

    if not candidates:
        return False, None

    r0, c0, marker_color = candidates[int(rng.integers(len(candidates)))]
    glyph_color, mask = GLYPHS[marker_color]
    grid[r0 : r0 + 4, c0 : c0 + 4][mask] = glyph_color
    grid[r0 + 4, c0 + 4] = marker_color
    return True, (r0, c0, marker_color)


def _build_case(
    num_types: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    used_types = MARKER_ORDER[:num_types]
    for _attempt in range(200):
        size = 20 if num_types >= 3 or rng.random() < 0.6 else 10
        row_positions, col_positions = _pick_layout(size, num_types, rng)
        markers = _choose_filled_cells(row_positions, col_positions, used_types, rng)

        expected = _render_output((size, size), markers)
        inp = np.zeros((size, size), dtype=int)
        for r, c, marker_color in markers:
            inp[r, c] = marker_color

        legend_present, legend_info = _place_legend(inp, used_types, rng)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue

        color_count = len({marker_color for _r, _c, marker_color in markers})
        if color_count != num_types:
            continue

        clipped_markers = sum(1 for r, c, _ in markers if r > size - 4 or c > size - 4)
        meta = {
            "num_types": num_types,
            "used_types": used_types,
            "marker_count": len(markers),
            "clipped_markers": clipped_markers,
            "legend_present": legend_present,
            "legend_info": legend_info,
            "grid_shape": inp.shape,
        }
        return inp, expected, meta

    raise RuntimeError(f"failed to build valid case with {num_types} marker types")


def generate_task(seed: int | None = None) -> dict[str, Any]:
    """
    Build a full D12 task with task-level type-count progression:
    train uses 2 then 3 marker colors, and test uses 4.
    """
    rng = np.random.default_rng(seed)
    train: list[dict[str, Any]] = []
    per_case_meta: list[dict[str, Any]] = []

    for num_types in TASK_TYPE_COUNTS[:-1]:
        inp, out, meta = _build_case(num_types, rng)
        train.append({"input": inp, "output": out})
        per_case_meta.append(meta)

    test_inp, test_out, test_meta = _build_case(TASK_TYPE_COUNTS[-1], rng)
    per_case_meta.append(test_meta)

    counts = [meta["num_types"] for meta in per_case_meta]
    assert counts == list(TASK_TYPE_COUNTS)
    return {
        "train": train,
        "test": [{"input": test_inp, "output": test_out}],
        "meta": {
            "marker_type_counts": counts,
            "used_type_prefixes": [meta["used_types"] for meta in per_case_meta],
            "legend_present": [meta["legend_present"] for meta in per_case_meta],
            "clipped_markers": [meta["clipped_markers"] for meta in per_case_meta],
        },
    }


def generate_input() -> np.ndarray:
    """
    Return the held-out test input from a task whose example sequence uses 2, then 3, then 4 marker colors.
    """
    task = generate_task()
    return np.array(task["test"][0]["input"], dtype=int)
