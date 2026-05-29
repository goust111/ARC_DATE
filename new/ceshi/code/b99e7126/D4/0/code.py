# Variant of b99e7126
# Dimension: D4 — 变换：分形缩放的方式 (transform)
# Change: 将异常单元格的NxN内部图案放大，使每个像素对应原网格中一个NxN单元格大小的区域... → 递归分形缩放：放大后的图案中再嵌套原始图案

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis
# ===============
# spec:
#   concretization:
#     - 保留“分隔线 + 规则单元格网格 + 多数/少数两类单元格图案”的基本骨架。
#     - 单元格内部仍是 3x3，少数单元格内部的 3x3 二值图案作为基础掩码 M。
#     - 原题输出是在大网格上放一个 3x3 的一级块；本变体改为两级递归：
#         level1 = M
#         level2 = 在 level1 的每个活跃格内再嵌入一份 M
#       等价于 9x9 递归掩码 kron(M, M)。
#     - 为了让 9x9 宏块能在网格中平移放置，单元格网格从 7x7 扩成 13x13；
#       像素网格变成 53x53 (= 13*4+1)。
#   behavior_trace:
#     - 原逻辑: 若少数单元格内 M 有 6 个活跃像素，则输出只补全一个 3x3 宏块中的 6 个少数单元格。
#     - 新逻辑: 同样的 M 先决定 3x3 一级布局，再让每个一级活跃格各自展开成一份 M，
#       所以最终少数单元格数变成 6*6=36，形成 9x9 的自相似结构。
#     - 输入中不直接给出全部 36 个位置，而是给出 2-3 个“完整子拷贝”加少量锚点，
#       让观察者能看到“单元格内部图案正在大尺度上重复”。
#   complexity_calibration:
#     - 基础掩码限定为 3x3、活跃数 5 或 6，避免 9x9 输出过密。
#     - 掩码要求四边都有活跃像素，且避免镜像/旋转对称，降低放置歧义。
#     - 输入显式展示 2-3 个完整二级子拷贝，既可见又不会直接把全输出泄露完。
#   implicit_assumption_audit:
#     - 原 `mh, mw = mask.shape` 默认“最终宏块大小 == 基础掩码大小”；这里不再成立，
#       必须改成 `fractal_mask.shape`。
#     - 原候选放置检查默认现有少数单元格只分布在一个 3x3 块内；递归后它们可能分布在 9x9
#       的多个子块里，因此匹配对象必须从 `mask` 换成 `kron(mask, mask)`。
#     - 原填充循环按 `mask[i, j]` 直接写多数/少数单元格；递归后要按 9x9 的二级掩码填充。
#     - `active_color = minority pattern 内最高频颜色` 仍可用，但生成器需保证活跃数 > 4。
#   placement_legality:
#     - 9x9 递归块放在 13x13 单元格网格内，平移范围是 0..4，不越界。
#     - 输出仍只含两种单元格图案；少数单元格都是完整 3x3 单元，不存在半格覆盖。
#   observability:
#     - 人眼可先在单个少数单元格里看到 3x3 掩码，再在输入中看到同样形状作为“完整子拷贝”
#       出现在更大尺度上，能归纳出递归复制而不是一次性放大。
#   visual_cleanliness:
#     - 仍只有 divider / 多数图案 / 少数图案 三层结构，没有额外标记层叠。
#     - 递归块被限制在一个 9x9 方窗内，整体整洁。
#   color_inference:
#     - 只使用 4 色：分隔线、majority 专属色、shared 色、minority 专属色。
#     - 颜色不参与排序或算术；active/inactive 完全由图案位置决定。
#   solver_generator_contract:
#     - solver 新增假设：输入里现有少数单元格必须是二级分形掩码的真子集，且其放置唯一。
#     - generator 通过“完整子拷贝 + 唯一候选校验 + `main(inp)==out` 验证”来保证这些假设。


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


def _recursive_fractal_mask(mask: np.ndarray, levels: int = 2) -> np.ndarray:
    """Repeat a boolean mask inside each active cell for the requested recursion depth."""
    out = mask.astype(int)
    base = mask.astype(int)
    for _ in range(levels - 1):
        out = np.kron(out, base)
    return out.astype(bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a recursive two-level fractal whose blueprint is encoded inside the minority cell.

    Input structure:
    - The grid is divided into equal cells by full separator rows and columns.
    - Most cells share one 3x3 interior pattern, while a minority pattern appears in several cells.
    - The minority cell's own 3x3 interior is the base blueprint for a larger recursive construction.

    Rule:
    - Read the minority cell's dominant-color mask as a 3x3 base pattern.
    - Expand that mask once to the cell-grid scale, then expand each active macro-cell again by embedding the same 3x3 mask inside it.
    - Find the unique 9x9 placement whose active positions are consistent with the minority cells already shown.
    - Fill every active second-level position with the minority pattern and every inactive position inside the placed 9x9 window with the majority pattern.

    Key insight: The odd cell does not merely say which macro-cells should differ once; it describes a pattern that recursively reappears inside each active macro-cell, so the large arrangement is a two-level self-copy of the same 3x3 motif.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(pat_arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)

    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            p = cell_pat[i][j]
            counts[p] = counts.get(p, 0) + 1
    assert len(counts) == 2, "must have exactly 2 distinct cell patterns (majority + minority)"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    base_mask = minority_arr == active_color
    fractal_mask = _recursive_fractal_mask(base_mask, levels=2)
    fh, fw = fractal_mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat

    candidates: list[tuple[int, int]] = []
    for top in range(0, n - fh + 1):
        for left in range(0, n - fw + 1):
            block = np.zeros_like(existing)
            block[top : top + fh, left : left + fw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + fh, left : left + fw]
            if np.any(ex_block & ~fractal_mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "must have a unique recursive mask placement"
    top, left = candidates[0]

    out = g.copy()
    for i in range(fh):
        for j in range(fw):
            target_pat = minority_arr if fractal_mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (recursive_fractal_tiling) ---------------

CELL_GRID_N = 13
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
MASK_SIZE = CELL_INNER
FRACTAL_LEVELS = 2
FRACTAL_SIZE = MASK_SIZE ** FRACTAL_LEVELS
PLACEMENT_RANGE = CELL_GRID_N - FRACTAL_SIZE + 1
ACTIVE_PIXEL_COUNTS = (5, 6)
GIVEN_MACRO_COPY_COUNTS = (2, 3)
COLOR_POOL = 10
MAX_RETRIES = 500


def _is_visibly_asymmetric(mask: np.ndarray) -> bool:
    """Reject masks whose large-scale placement would look too symmetric or ambiguous."""
    return not any(
        np.array_equal(mask, transformed)
        for transformed in (
            np.fliplr(mask),
            np.flipud(mask),
            np.rot90(mask, 1),
            np.rot90(mask, 2),
            np.rot90(mask, 3),
        )
    )


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate 3x3 base masks that stay legible after recursive expansion."""
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        masks: list[np.ndarray] = []
        for combo in combinations(range(MASK_SIZE * MASK_SIZE), active_count):
            mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
            mask.flat[list(combo)] = True
            if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
                continue
            if not _is_visibly_asymmetric(mask):
                continue
            masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_two_color_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform 3x3 pattern that uses both colors."""
    arr = np.array([random.choice((color_a, color_b)) for _ in range(MASK_SIZE * MASK_SIZE)], dtype=int).reshape(MASK_SIZE, MASK_SIZE)
    if np.all(arr == arr[0, 0]):
        arr[random.randrange(MASK_SIZE), random.randrange(MASK_SIZE)] = color_b if arr[0, 0] == color_a else color_a
    if not np.any(arr == color_a):
        arr[0, 0] = color_a
    if not np.any(arr == color_b):
        arr[-1, -1] = color_b
    return arr


def _random_minority_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count]).copy()


def _relative_active_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(coord) for coord in np.argwhere(mask)]


def _fractal_subset_for_macro_positions(base_mask: np.ndarray, macro_subset: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    """Reveal full second-level copies for the chosen active first-level macro-cells."""
    mh, mw = base_mask.shape
    micro_active = _relative_active_positions(base_mask)
    subset: set[tuple[int, int]] = set()
    for mr, mc in macro_subset:
        for r, c in micro_active:
            subset.add((mr * mh + r, mc * mw + c))
    return sorted(subset)


def _placement_candidates(fractal_mask: np.ndarray, subset: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    """Find all placements where the revealed second-level cells are consistent with the recursive mask."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True

    fh, fw = fractal_mask.shape
    candidates: list[tuple[int, int]] = []
    for t in range(PLACEMENT_RANGE):
        for l in range(PLACEMENT_RANGE):
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[t : t + fh, l : l + fw] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[t : t + fh, l : l + fw] & ~fractal_mask):
                continue
            candidates.append((t, l))
    return candidates


def _choose_given_subset(base_mask: np.ndarray, fractal_mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]] | None:
    """
    Pick a visible subset:
    - start with 2-3 full second-level copies so recursion is obvious
    - optionally add 1-2 anchor cells if needed to make the placement unique
    """
    macro_active = _relative_active_positions(base_mask)
    fractal_active = _relative_active_positions(fractal_mask)

    sizes = list(GIVEN_MACRO_COPY_COUNTS)
    random.shuffle(sizes)
    for size in sizes:
        if size >= len(macro_active):
            continue
        macro_combos = list(combinations(macro_active, size))
        random.shuffle(macro_combos)
        for macro_subset in macro_combos:
            base_subset = set(_fractal_subset_for_macro_positions(base_mask, macro_subset))
            remaining = [pos for pos in fractal_active if pos not in base_subset]

            extra_options: list[tuple[tuple[int, int], ...]] = [tuple()]
            extra_options.extend((pos,) for pos in remaining)
            if len(remaining) >= 2:
                pair_pool = list(combinations(remaining, 2))
                random.shuffle(pair_pool)
                extra_options.extend(pair_pool[:16])

            for extras in extra_options:
                subset = sorted(base_subset | set(extras))
                if _placement_candidates(fractal_mask, subset, top, left) == [(top, left)]:
                    return subset
    return None


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, minority: np.ndarray, active_cells: list[tuple[int, int]]) -> np.ndarray:
    """Construct the full grid by stamping either the majority or minority cell pattern."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_two_color_pattern(shared, maj_exclusive)
    base_mask = _random_minority_mask()
    minority = np.where(base_mask, min_exclusive, shared).astype(int)
    fractal_mask = _recursive_fractal_mask(base_mask, levels=FRACTAL_LEVELS)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + r, left + c) for r, c in _relative_active_positions(fractal_mask)]
    out = _build_grid(separator, majority, minority, active_cells)

    subset = _choose_given_subset(base_mask, fractal_mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, minority, given_cells)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid recursive-fractal b99e7126 input")
