# Combo variant of b99e7126 (index 54)
# combo_spec: D3/2+D10/1
# Name: 先叠加后提取 + 锚点标记
#
# Design analysis
# ---------------
# variant:
#   reference_baseline:
#     original_b99e7126:
#       - 画面是 7x7 的规则单元格网格，每个单元内部都是 3x3 图案。
#       - 少数格的 3x3 图案本身是一个网格级 3x3 mask，原题要把这个 mask 放大成 3x3 个单元并覆盖回整张图。
#       - 原题靠已有 minority 格的绝对位置去反推放大块左上角，输出是整张覆盖后的大图。
#     D3_2_marker_alignment:
#       - 额外加入一个 marker 格，视觉形式是 majority 图案中央 1 像素改成独有颜色。
#       - marker 不提供模板，只提供几何锚点；真正的模板仍来自 minority 格。
#     D10_1_overlay_then_extract:
#       - 放大叠加只作为中间步骤，最终输出不是整图，而是从叠加结果里截取一个局部窗口。
#   concrete_visualization:
#     majority_cell:
#       - 仍是大面积重复的 3x3 双色对称图案，作为稳定背景。
#     minority_cell:
#       - 仍是另一种 3x3 双色图案；其高频色定义 3x3 active mask。
#       - 生成器强制中心像素 active，这样 marker 作为中心锚点时会在中间步骤里被 minority 覆盖。
#     marker_cell:
#       - 与 majority 只差中心 1 像素，且全图唯一。
#       - 肉眼可看成“这里是提示中心”，与真正 minority 格明显不是同一类对象。
#   behavior_trace:
#     small_example:
#       - 假设 minority 的内部 mask 在 3x3 中有 6 个 active，输入只给出其中 2 个 minority 格，另有 1 个带中心点的 marker 格。
#       - 原题会用那 2 个 minority 格去穷举放大块位置，再返回整张 29x29 覆盖图。
#       - 本融合版先读出 minority 模板，再用 marker 定位 3x3 放大块中心，完成叠加后，只截取 marker 周围半径 2 个单元的窗口。
#       - 如果 marker 靠近整图边缘，窗口会被边界裁切，因此不同锚点会导出不同尺寸/不同边界关系的输出。
#     key_difference:
#       - D3/2 改掉了“最少见模式就是 minority 且由 minority 反推位置”的旧假设。
#       - D10/1 改掉了“solver 最终直接返回整张覆盖图”的旧假设。
#       - 两者联动后的关键是：marker 同时控制“叠加在哪”与“最终裁哪一块”；若只保留任一部分，规则都会退化。
#   complexity_calibration:
#     - 颜色从 4 种增到 5 种，只新增 1 个 marker 色，视觉负担仍低。
#     - minority 提示格只保留 2-3 个，保证不靠旧规则唯一定位；marker 是必须信息。
#     - 提取窗口固定为“以 marker 为中心、半径 2 个单元，再按整图边界裁切”，只有一种后处理档位，避免规则过散。
#   implicit_assumption_audit:
#     original_main_lines_that_break:
#       - `min(counts)` 原先默认最少见图案就是 minority；现在最少见的是 marker。
#       - `existing == minority_pat` 原先默认所有特殊格都同类；现在必须先把 marker 和 minority 分开。
#       - 穷举 placement 的那段逻辑原先依赖“已有 minority 子集足够唯一”；现在 generator 反而要保证忽略 marker 时通常不唯一。
#       - 直接 `return out` 原先默认输出整图；现在必须先生成中间叠加结果，再做中心窗口裁切。
#     new_assumptions_and_generator_support:
#       - marker 图案与 majority 恰好汉明距离 1 且只出现 1 次；generator 直接构造保证。
#       - minority 与 majority 的距离至少 3，防止被误判成 marker。
#       - minority mask 的中心像素 active；预计算 mask 时强制成立。
#       - 输出窗口是“中心半径 2”并允许碰边裁切；因此 top/left 允许回到原题的 0..4，全域都合法。
#   placement_legality:
#     - 放大块仍是 3x3 个单元，只需保证其左上角在 0..4。
#     - marker 位于该 3x3 放大块中心，即单元坐标 `(top+1, left+1)`。
#     - 输入 minority 提示格只从 active 位置中选，并排除中心 anchor，避免与 marker 重叠。
#     - 裁切窗口由 marker 决定并对整图边界做截断，因此不会越界。
#   observability_and_cleanliness:
#     - 输入里能直接看见三层结构：大面积重复 majority、少量真正 minority、以及唯一带中心点的 marker。
#     - 输出尺寸可能因 marker 靠边而变化，这让“先定位、再从该位置取窗口”的后处理更可观察，不会退化成平移不变的小块。
#     - 输出中 marker 会被 minority 覆盖，强化“marker 只是定位提示，不是最终保留对象”的因果关系。
#   inference_guardrails:
#     - 不依赖颜色编号大小；所有判断都来自图案频次、像素差异和几何位置。
#     - generator 会拒绝“忽略 marker 也导出同一输出”的样本，确保融合行为真实发生。

from __future__ import annotations

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    """
    Infer a regular cell grid separated by full rows/cols of the same line color.
    Returns (step, row_divs, col_divs) where:
      - row_divs/col_divs are divider indices
      - step is the spacing between dividers (cell size including divider).
    """
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need at least 2 row and 2 col dividers"
    row_divs.sort()
    col_divs.sort()
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1) if row_divs[i + 1] > row_divs[i]]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1) if col_divs[i + 1] > col_divs[i]]
    step = reduce(gcd, drs + dcs)
    assert step > 1, "divider step must be > 1"
    assert row_divs[0] == 0 and col_divs[0] == 0, "first divider must be at index 0"
    assert (h - 1) % step == 0 and (w - 1) % step == 0, "grid dimensions must be (k*step + 1)"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "last divider must be at boundary"
    return int(step), row_divs, col_divs


def _extract_cell_patterns(
    g: np.ndarray,
) -> tuple[int, int, dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]], dict[tuple[int, ...], int]]:
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(pat_arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = pat_arr
            counts[pat] = counts.get(pat, 0) + 1
            row.append(pat)
        cell_pat.append(row)
    return step, n, patterns, cell_pat, counts


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _find_pattern_cells(cell_pat: list[list[tuple[int, ...]]], target: tuple[int, ...]) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    for i, row in enumerate(cell_pat):
        for j, pat in enumerate(row):
            if pat == target:
                hits.append((int(i), int(j)))
    return hits


def _active_mask_from_minority(minority_arr: np.ndarray) -> tuple[np.ndarray, int]:
    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    return mask, active_color


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _apply_overlay(
    grid: np.ndarray,
    majority: np.ndarray,
    minority: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
) -> np.ndarray:
    out = grid.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            target = minority if mask[i, j] else majority
            paste(out, target, *_cell_origin(top + i, left + j))
    return out


def _extract_centered_window(grid: np.ndarray, center_cell: tuple[int, int], radius_cells: int) -> np.ndarray:
    center_r, center_c = center_cell
    top_cell = max(0, center_r - radius_cells)
    left_cell = max(0, center_c - radius_cells)
    bottom_cell = min(CELL_GRID_N - 1, center_r + radius_cells)
    right_cell = min(CELL_GRID_N - 1, center_c + radius_cells)

    r0 = top_cell * CELL_STEP
    c0 = left_cell * CELL_STEP
    r1 = (bottom_cell + 1) * CELL_STEP + 1
    c1 = (right_cell + 1) * CELL_STEP + 1
    return grid[r0:r1, c0:c1].copy()


def _count_changed_cells(a: np.ndarray, b: np.ndarray) -> int:
    step_a, _, _, cells_a, _ = _extract_cell_patterns(a)
    step_b, _, _, cells_b, _ = _extract_cell_patterns(b)
    assert step_a == step_b == CELL_STEP
    assert len(cells_a) == len(cells_b) and len(cells_a[0]) == len(cells_b[0])

    changed = 0
    for i in range(len(cells_a)):
        for j in range(len(cells_a[0])):
            if cells_a[i][j] != cells_b[i][j]:
                changed += 1
    return int(changed)


def _serialize_grid(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, grid.shape)), tuple(map(int, grid.ravel().tolist()))


def _old_style_candidates(mask: np.ndarray, given_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in given_cells:
        existing[r, c] = True

    candidates: list[tuple[int, int]] = []
    for top in range(PLACEMENT_RANGE):
        for left in range(PLACEMENT_RANGE):
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[top : top + MASK_SIZE, left : left + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + MASK_SIZE, left : left + MASK_SIZE]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    return candidates


def _overlay_then_extract(
    base_grid: np.ndarray,
    majority: np.ndarray,
    minority: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
) -> np.ndarray:
    overlaid = _apply_overlay(base_grid, majority, minority, mask, top, left)
    marker_cell = (top + ANCHOR_REL[0], left + ANCHOR_REL[1])
    return _extract_centered_window(overlaid, marker_cell, CROP_RADIUS_CELLS)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the anomaly cell as a fractal template, stamp it by a separate marker anchor, then return only the anchored local window.

    Input structure:
    - The picture is a 7x7 grid of 3x3 cell interiors separated by 1-pixel divider lines.
    - One interior pattern is the repeated majority background.
    - A few cells show one distinct minority pattern, and one extra marker cell matches the majority except for a single special center pixel.

    Rule:
    - Identify the minority pattern and read its dominant color as the active pixels of a 3x3 mask.
    - Identify the marker cell and treat it as the center anchor of the enlarged 3x3 cell-level overlay.
    - First overwrite that anchored 3x3 cell block: active mask positions become minority cells, inactive positions become majority cells.
    - Then crop the overlaid result to the window centered on the marker with radius 2 cells, clipping against the full 7x7 canvas edges when necessary.

    Key insight:
    - The minority cell tells you what to stamp, while the one-pixel marker tells you both where to stamp it and which neighborhood to keep.
    - Because the final answer is a cropped post-overlay view rather than the whole canvas, the solver must perform the overlay step before extracting the output.
    """
    g = np.asarray(grid, dtype=int)
    step, n, patterns, cell_pat, counts = _extract_cell_patterns(g)
    assert step == CELL_STEP and n == CELL_GRID_N, "unexpected geometry"
    assert len(counts) >= 3, "need majority, minority, and marker patterns"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    marker_candidates: list[tuple[int, ...]] = []
    minority_candidates: list[tuple[int, ...]] = []
    for pat, cnt in counts.items():
        if pat == majority_pat:
            continue
        arr = patterns[pat]
        if cnt == 1 and _hamming_distance(arr, majority_arr) == 1:
            marker_candidates.append(pat)
        else:
            minority_candidates.append(pat)

    assert len(marker_candidates) == 1, "must have exactly one marker pattern"
    assert len(minority_candidates) == 1, "must have exactly one minority pattern"

    marker_pat = marker_candidates[0]
    minority_pat = minority_candidates[0]
    minority_arr = patterns[minority_pat]
    mask, _ = _active_mask_from_minority(minority_arr)
    anchor_r = mask.shape[0] // 2
    anchor_c = mask.shape[1] // 2
    assert mask[anchor_r, anchor_c], "center anchor must be active in the minority mask"

    marker_cells = _find_pattern_cells(cell_pat, marker_pat)
    minority_cells = _find_pattern_cells(cell_pat, minority_pat)
    assert len(marker_cells) == 1, "marker must occur exactly once"
    assert minority_cells, "input must contain at least one minority hint"

    marker_i, marker_j = marker_cells[0]
    top = marker_i - anchor_r
    left = marker_j - anchor_c
    assert 0 <= top <= n - mask.shape[0] and 0 <= left <= n - mask.shape[1], "marker anchor would place block out of bounds"

    for cell_i, cell_j in minority_cells:
        rel_i = cell_i - top
        rel_j = cell_j - left
        assert 0 <= rel_i < mask.shape[0] and 0 <= rel_j < mask.shape[1], "minority hint falls outside anchored block"
        assert mask[rel_i, rel_j], "minority hint must land on an active mask position"

    return _overlay_then_extract(g, majority_arr, minority_arr, mask, top, left)


# --------------- generator (o5x) ---------------

# --- CFG: shared geometry ---
CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ANCHOR_REL = (1, 1)
CROP_RADIUS_CELLS = 2

# --- CFG: pattern / color choices ---
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [2, 3]
COLOR_POOL = 10
MAX_RETRIES = 600
MIN_MINORITY_DISTANCE = 3
MIN_CROP_DELTA_CELLS = 3
MIN_ALT_OUTPUTS = 2

# LR-symmetric pixel groups: pairs share left/right, singletons sit on center col.
_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate valid LR-symmetric 3x3 masks with active center and edge support."""
    weights = [len(group) for group in _LR_GROUPS]
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
        for k in range(1, len(_LR_GROUPS) + 1):
            for combo in combinations(range(len(_LR_GROUPS)), k):
                if sum(weights[i] for i in combo) != inactive_count:
                    continue
                mask = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
                for idx in combo:
                    for r, c in _LR_GROUPS[idx]:
                        mask[r, c] = False
                if not mask[ANCHOR_REL]:
                    continue
                if mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a random LR-symmetric 3x3 pattern using two colors (non-uniform)."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _build_grid(
    separator: int,
    majority: np.ndarray,
    minority_cells: list[tuple[int, int]],
    minority: np.ndarray,
    marker_cell: tuple[int, int] | None = None,
    marker_pattern: np.ndarray | None = None,
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in minority_cells:
        paste(grid, minority, *_cell_origin(i, j))
    if marker_cell is not None and marker_pattern is not None:
        paste(grid, marker_pattern, *_cell_origin(*marker_cell))
    return grid


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]] | None:
    """Pick active minority hints that stay ambiguous under the old no-marker placement rule."""
    active = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    active = [coord for coord in active if coord != ANCHOR_REL]
    random.shuffle(active)
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid: list[list[tuple[int, int]]] = []
        for subset in combinations(active, size):
            given_cells = [(top + r, left + c) for r, c in subset]
            candidates = _old_style_candidates(mask, given_cells)
            if len(candidates) > 1 and (top, left) in candidates:
                valid.append(given_cells)
        if valid:
            return random.choice(valid)
    return None


def _analyze_variant_input(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify the non-degenerate fused behavior on one generated input.

    Metrics:
      - old_candidate_count: how many placements remain if marker is ignored
      - old_unique_output_count: how many distinct cropped outputs those old placements induce
      - crop_delta_cells: how many cell interiors differ between direct input-crop and true output
      - output_h/output_w: cropped output size in pixels
    """
    g = np.asarray(grid, dtype=int)
    step, n, patterns, cell_pat, counts = _extract_cell_patterns(g)
    assert step == CELL_STEP and n == CELL_GRID_N

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    marker_pat = None
    minority_pat = None
    for pat, cnt in counts.items():
        if pat == majority_pat:
            continue
        arr = patterns[pat]
        if cnt == 1 and _hamming_distance(arr, majority_arr) == 1:
            marker_pat = pat
        else:
            minority_pat = pat
    assert marker_pat is not None and minority_pat is not None

    marker_cell = _find_pattern_cells(cell_pat, marker_pat)[0]
    minority_cells = _find_pattern_cells(cell_pat, minority_pat)
    minority_arr = patterns[minority_pat]
    mask, _ = _active_mask_from_minority(minority_arr)

    input_crop = _extract_centered_window(g, marker_cell, CROP_RADIUS_CELLS)
    expected = main(g)
    crop_delta_cells = _count_changed_cells(input_crop, expected)

    marker_removed = g.copy()
    paste(marker_removed, majority_arr, *_cell_origin(*marker_cell))
    old_candidates = _old_style_candidates(mask, minority_cells)

    old_outputs = set()
    for top, left in old_candidates:
        candidate = _overlay_then_extract(marker_removed, majority_arr, minority_arr, mask, top, left)
        old_outputs.add(_serialize_grid(candidate))

    return {
        "old_candidate_count": int(len(old_candidates)),
        "old_unique_output_count": int(len(old_outputs)),
        "crop_delta_cells": int(crop_delta_cells),
        "output_h": int(expected.shape[0]),
        "output_w": int(expected.shape[1]),
        "marker_row": int(marker_cell[0]),
        "marker_col": int(marker_cell[1]),
        "minority_count": int(len(minority_cells)),
    }


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, min_exclusive, marker_exclusive = random.sample(range(COLOR_POOL), 5)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)
    if _hamming_distance(minority, majority) < MIN_MINORITY_DISTANCE:
        return None

    marker_pattern = majority.copy()
    marker_pattern[ANCHOR_REL] = marker_exclusive
    if np.array_equal(marker_pattern, minority):
        return None

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    marker_cell = (top + ANCHOR_REL[0], left + ANCHOR_REL[1])
    active_cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]

    given_cells = _choose_given_subset(mask, top, left)
    if given_cells is None:
        return None

    inp = _build_grid(separator, majority, given_cells, minority, marker_cell=marker_cell, marker_pattern=marker_pattern)
    marker_removed = _build_grid(separator, majority, given_cells, minority)
    expected = _overlay_then_extract(marker_removed, majority, minority, mask, top, left)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    metrics = _analyze_variant_input(inp)
    if metrics["old_candidate_count"] <= 1:
        return None
    if metrics["old_unique_output_count"] < MIN_ALT_OUTPUTS:
        return None
    if metrics["crop_delta_cells"] < MIN_CROP_DELTA_CELLS:
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo D3/2+D10/1 input")
