# Variant of b99e7126
# Dimension: D3 — 目标位置：放大图案的叠加定位方式 (dst)
# Change: 放大后的图案定位使得源单元格位置与放大图案中对应的单元格重合，即以异常单元格为锚点对齐...
# → 锚点由另一个特殊标记单元格确定
#
# Design analysis
# ---------------
# variant:
#   reference_baseline:
#     original_b99e7126:
#       - 主体仍是 7x7 的单元格网格，每个单元内部是 3x3 图案。
#       - 少数单元的 3x3 内部图案本身就是网格级 3x3 放大遮罩。
#       - 原题用已有 minority 单元的位置来反推 3x3 放大块该放在哪。
#     source_task_581f7754:
#       - 其核心复杂度不在对象变形，而在"识别另一类标记，再用标记决定对齐位置"。
#       - 因此本变体也只新增一个显式 marker，并让它控制放大块锚点，不引入额外几何变换。
#   concrete_visualization:
#     majority_cell:
#       - 仍是重复出现的 3x3 双色图案。
#     minority_cell:
#       - 仍是另一种 3x3 双色图案；高频颜色像素定义网格级 active mask。
#       - 为了让"center anchor"有意义，生成器强制 mask 的中心像素为 active。
#     marker_cell:
#       - 视觉上接近 majority，但中心 1 个像素被替换成独有 marker 色。
#       - 这让人能把它看成"带一个点的提示格"，并且和 minority 明显区分。
#   behavior_trace:
#     original_logic:
#       - 根据输入里已有 minority 单元的绝对位置，枚举 3x3 放大块左上角，找唯一可行位置。
#     new_logic:
#       - 先识别 minority 模板，再识别 marker 单元。
#       - marker 单元固定对应放大块中心，因此 left/top = marker_cell - (1,1)。
#       - 最后按 minority 的内部 mask，在这个由 marker 指定的 3x3 区域内完成覆盖。
#     key_difference:
#       - 旧逻辑依赖"已有 minority 单元足够唯一"；新逻辑依赖"marker 给锚点"。
#       - 生成器显式保证：忽略 marker 时，已有 minority 子集会对应多个合法旧式 placement。
#   complexity_calibration:
#     - 颜色从 4 种增到 5 种，只多一个 marker 色，复杂度仍然低。
#     - minority 仍保留 2-3 个已给提示格，避免任务退化成"只看一个单元然后机械复制"。
#     - marker 只有一种形式：majority + center dot，避免引入多档规则。
#   implicit_assumption_audit:
#     original_main_lines_that_break:
#       - "minority_pat = min(counts...)" 原先默认最少见图案就是异常图案；现在最少见的会是 marker。
#       - "existing[i, j] = cell_pat[i][j] == minority_pat" 原先默认所有特殊格都是同一种；现在必须区分 marker 和 minority。
#       - placement candidate search 原先默认锚点来自已有 minority 的全局位置；现在锚点来自 marker，不能再靠穷举已有 minority 反推。
#     new_assumptions_and_generator_support:
#       - marker 图案与 majority 恰好只差 1 个像素，且只出现 1 次；generator 直接构造保证。
#       - minority 图案与 majority 不会只差 1 个像素；generator 若距离太小会拒绝。
#       - mask 中心像素为 active；mask 预计算时强制成立。
#       - marker 位于可容纳 3x3 放大块中心的位置，即单元坐标在 1..5；generator 用 top/left 0..4 保证。
#   placement_legality:
#     - 放大块仍是 3x3 个单元，因此只需保证 marker 在中心单元合法，等价于 top/left 在 0..4。
#     - 输入里的 minority 提示格只从该放大块的 active 位置中选，且排除中心 marker 格，避免图案冲突。
#   observability_and_cleanliness:
#     - 输入里能直接看到两类特殊格：若干真正的 minority 格，以及一个"多数格中间点了新颜色"的 marker 格。
#     - 输出里 marker 会被真正的 minority 格覆盖，形成"marker 指定中心，周围按模板长出来"的可见因果关系。
#     - 没有使用颜色 id 排序或不可见属性；所有判断都来自可见的像素结构和相对位置。

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


def _extract_cell_patterns(g: np.ndarray) -> tuple[int, int, dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]], dict[tuple[int, ...], int]]:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the anomaly cell's interior pattern onto the cell grid, but anchor the overlay with a separate marker cell.

    Input structure:
    - The canvas is a regular NxN grid of 3x3 cell interiors separated by 1-pixel divider lines.
    - Most cells share one majority interior pattern.
    - A few cells use one distinct minority pattern, and one extra marker cell looks like the majority cell except for a single special center pixel.

    Rule:
    - Read the minority cell's 3x3 interior as a boolean mask using its dominant color as "active".
    - Treat each active mask pixel as a request to place a full minority cell, and each inactive pixel as a majority cell.
    - Use the marker cell, not the minority cells themselves, to anchor the enlarged 3x3 placement: the marker cell is the center of the enlarged block.
    - Replace that 3x3 block accordingly, turning the marker cell itself into whichever cell type the centered mask pixel demands.

    Key insight:
    - Two different special cell types must be separated first: the true anomaly template and the one-pixel marker.
    - The anomaly provides the fractal blueprint, while the marker alone fixes where that blueprint is stamped onto the larger cell grid.
    """
    g = np.asarray(grid, dtype=int)
    step, n, patterns, cell_pat, counts = _extract_cell_patterns(g)
    inner = step - 1
    assert len(counts) >= 3, "need majority, minority, and marker patterns"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    marker_candidates: list[tuple[int, ...]] = []
    minority_candidates: list[tuple[int, ...]] = []
    for pat, cnt in counts.items():
        if pat == majority_pat:
            continue
        arr = patterns[pat]
        dist = _hamming_distance(arr, majority_arr)
        if dist == 1 and cnt == 1:
            marker_candidates.append(pat)
        else:
            minority_candidates.append(pat)

    assert len(marker_candidates) == 1, "must have exactly one marker pattern"
    assert len(minority_candidates) == 1, "must have exactly one minority pattern"

    marker_pat = marker_candidates[0]
    minority_pat = minority_candidates[0]
    minority_arr = patterns[minority_pat]
    mask, _active_color = _active_mask_from_minority(minority_arr)
    mh, mw = mask.shape
    anchor_r = mh // 2
    anchor_c = mw // 2
    assert mask[anchor_r, anchor_c], "center anchor must be active in the minority mask"

    marker_cells = _find_pattern_cells(cell_pat, marker_pat)
    assert len(marker_cells) == 1, "marker must occur exactly once"
    marker_i, marker_j = marker_cells[0]
    top = marker_i - anchor_r
    left = marker_j - anchor_c
    assert 0 <= top <= n - mh and 0 <= left <= n - mw, "marker anchor would place block out of bounds"

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target_pat = minority_arr if mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (o5x) ---------------

# --- CFG: shared geometry ---
CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ANCHOR_REL = (1, 1)

# --- CFG: pattern / color choices ---
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 2]
COLOR_POOL = 10
MAX_RETRIES = 400
MIN_MINORITY_DISTANCE = 3

# LR-symmetric pixel groups: pairs share left/right, singletons sit on center col.
_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _precompute_masks():
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


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _old_style_candidates(mask: np.ndarray, given_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Placement candidates under the original rule that ignores the marker cell."""
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


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]] | None:
    """Pick active minority hints that remain ambiguous without the marker anchor."""
    active = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    active = [coord for coord in active if coord != ANCHOR_REL]
    random.shuffle(active)
    for size in GIVEN_CELL_SIZES:
        valid: list[list[tuple[int, int]]] = []
        for subset in combinations(active, size):
            given_cells = [(top + r, left + c) for r, c in subset]
            candidates = _old_style_candidates(mask, given_cells)
            if len(candidates) > 1 and (top, left) in candidates:
                valid.append(given_cells)
        if valid:
            return random.choice(valid)
    return None


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


def _analyze_variant_input(grid: np.ndarray) -> dict[str, int]:
    """Metric helper for post-test validation: old rule must stay ambiguous."""
    g = np.asarray(grid, dtype=int)
    _step, _n, patterns, cell_pat, counts = _extract_cell_patterns(g)
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    marker_pat = None
    minority_pat = None
    for pat, cnt in counts.items():
        if pat == majority_pat:
            continue
        arr = patterns[pat]
        if _hamming_distance(arr, majority_arr) == 1 and cnt == 1:
            marker_pat = pat
        else:
            minority_pat = pat
    assert marker_pat is not None and minority_pat is not None

    marker_cell = _find_pattern_cells(cell_pat, marker_pat)[0]
    minority_cells = _find_pattern_cells(cell_pat, minority_pat)
    mask, _ = _active_mask_from_minority(patterns[minority_pat])
    old_candidates = _old_style_candidates(mask, minority_cells)
    return {
        "minority_count": int(len(minority_cells)),
        "old_candidate_count": int(len(old_candidates)),
        "marker_row": int(marker_cell[0]),
        "marker_col": int(marker_cell[1]),
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

    out = _build_grid(separator, majority, active_cells, minority)
    inp = _build_grid(separator, majority, given_cells, minority, marker_cell=marker_cell, marker_pattern=marker_pattern)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    metrics = _analyze_variant_input(inp)
    if metrics["old_candidate_count"] <= 1:
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 D3 marker-anchor input")
