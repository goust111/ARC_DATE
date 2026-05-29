# Variant of b99e7126
# Dimension: D3 — 目标位置：放大图案的叠加定位方式 (dst)
# Change: 放大后的图案定位使得源单元格位置与放大图案中对应的单元格重合，即以异常单元格为锚点对齐... → 多次叠加：图案在多个位置重复放置
#
# Design analysis:
# variant_spec:
#   core_change:
#     original_dst: 在 7x7 单元格网格中找到唯一一个 3x3 放大块的位置，然后只叠加这一次。
#     new_dst: 先用输入中的少数单元锁定一个合法 3x3 放大块列位置与行奇偶类；随后把同一放大块
#       沿竖直方向以 2 个单元格为步长重复放置到所有合法 top。多个放大块之间会重叠 1 行，
#       输出取所有放大块 active 像素的并集；并集为 True 的单元格写入 minority pattern，否则写 majority pattern。
#   concrete_behavior:
#     - 网格仍是 29x29，由 7x7 个 3x3 interior 单元格和 1 像素分隔线构成。
#     - minority 单元内部的 3x3 active mask 仍来自其高频色像素。
#     - 输入里仍只给出 3-4 个 minority 单元，而且这些已知 minority 都来自同一个 3x3 放大块实例。
#     - 输出里不只完成该实例，而是把这个 3x3 放大块在同一列位置上向上/向下每隔 2 行重复一次。
#   example_trace:
#     original:
#       - 设 mask active 位于局部坐标 {(0,0),(1,0),(1,1),(2,1)}，输入给了其中 3 个 minority cells。
#       - solver 找到唯一 top,left=(2,1)，只在 rows 2..4, cols 1..3 这一块补全剩余 active。
#     variant:
#       - 同样先找出给定 minority 可落在 top,left=(2,1) 的某个 3x3 块内。
#       - 但输出不是只补 rows 2..4，而是把相同 mask 再放到 top=0,2,4 且 left=1 的所有块。
#       - 这些块在 row 2 和 row 4 发生重叠；某格只要任一块对应 mask=True，就保留 minority。
#     difference:
#       - 原题的唯一性对象是单个 anchor 块。
#       - 变体的唯一性对象是“left 固定、top 奇偶固定”的重复类；同类里可有多个 anchor top，
#         但它们导出的最终输出相同。
#   complexity_calibration:
#     - 若改成在所有 25 个位置都重复放置，画面会过密且原规则痕迹太弱；若只放 2 次又不够明显。
#     - 竖直方向步长 2 的重复类恰好带来 2 或 3 个实例，视觉上明显，同时搜索空间仅 10 类
#       (left∈0..4, parity∈{0,1})，不至于过度自由。
#   implicit_assumptions_review:
#     - 原 main 中 `assert len(candidates) == 1` 依赖“输出只对应一个 anchor 块”。
#       变体下这不成立，因为 overlap 区域里的输入子集可能对应同一重复类中的多个 top；
#       新 solver 改为对 `(top % 2, left)` 去重，并要求重复类唯一。
#     - 原 main 的逐格写入默认每个网格单元只被一个放大块访问；变体下若仍按块顺序写 majority/minority，
#       后写入的 inactive 像素会错误覆盖先前的 active。新 solver 先做并集 mask，再一次性着色。
#     - 原 main 假设输入 minority 都落在唯一 3x3 块内。这个假设保留，因为 generator 仍从单个实例采样 givens。
#   placement_legality:
#     - 所有重复块的 top 来自 `range(parity, 5, 2)`，left 固定在 0..4，因此每个 3x3 块都完全在 7x7 网格内。
#     - 重叠只发生在相邻重复块之间的 1 行处，不会越界，也不会引入第三种单元 pattern。
#     - generator 通过 output-first 构造 output，再用 `main(input)` 校验，保证 solver/generator 一致。
#   visual_observability:
#     - 解题者可见：输入只有一小簇 minority cells；输出沿同一列扩张成重复的竖直带状结构。
#     - 重复步长 2 是空间关系，不依赖颜色 id 或隐藏排序；多组样例里 left 与 parity 随机变化，
#       但“同列竖直重复”这一规律稳定可见。
#   cleanliness:
#     - 只在一个 3 列宽的带状区域做多次叠加，避免全图杂乱。
#     - overlap 仅是并集，不引入多层颜色编码，最终每个单元仍只有 majority/minority 两类清晰纹理。
#   color_inference:
#     - 颜色角色仍只有 separator / majority-exclusive / shared / minority-exclusive 四种。
#     - 颜色不参与排序；唯一作用是 minority 高频色定义 active mask，与原题一致，易于从样例归纳。
#   solver_assumptions_after_change:
#     - 输入仍恰有两类 cell interior pattern，minority 是计数更少者。
#     - minority interior 的高频色定义 3x3 mask，且输入给定 minority 是该 mask 某个单实例的真子集。
#     - 在 10 个重复类 `(parity,left)` 中恰有 1 类与 givens 相容；generator 通过子集筛选保证这一点。

from functools import reduce
from math import gcd

import numpy as np
import random
from itertools import combinations
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


def _repeated_vertical_union(mask: np.ndarray, parity: int, left: int, n: int) -> np.ndarray:
    """Union of all mask placements in the same vertical repeat class."""
    mh, mw = mask.shape
    union = np.zeros((n, n), dtype=bool)
    for top in range(parity, n - mh + 1, REPEAT_STEP):
        union[top : top + mh, left : left + mw] |= mask
    return union


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a minority cell's inner pattern to the cell grid, then repeat that enlarged motif vertically.

    Input structure:
    - The grid is partitioned into equal cells by single-color divider lines.
    - Almost every cell shares one interior pattern, while a few cells use a distinct minority pattern.
    - The visible minority cells all come from one 3x3 enlarged motif instance.

    Rule:
    - Read the minority cell's 3x3 interior as a binary mask using its most frequent color as the active color.
    - Find which 3x3 cell block can contain the given minority cells, but only keep the unique repeat class
      defined by the block's column and row parity.
    - Copy that enlarged 3x3 motif to every valid block in the same column and row-parity class.
    - If repeated placements overlap, any active hit keeps the minority pattern; only cells untouched by all
      active mask pixels remain majority.

    Key insight: The odd cell still encodes the large-scale blueprint, but now that blueprint is reused as a
    vertically repeating stencil rather than a one-off placement.
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
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns (minority + majority)"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat

    placement_candidates: list[tuple[int, int]] = []
    for top in range(0, n - mh + 1):
        for left in range(0, n - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~mask):
                continue
            placement_candidates.append((int(top), int(left)))

    repeat_classes = sorted({(top % REPEAT_STEP, left) for top, left in placement_candidates})
    assert len(repeat_classes) == 1, "must have a unique vertical repeat class aligning the given minority cells"
    parity, left = repeat_classes[0]
    union = _repeated_vertical_union(mask, parity, left, n)

    out = g.copy()
    majority_arr = patterns[majority_pat]
    for i in range(n):
        for j in range(n):
            target_pat = minority_arr if union[i, j] else majority_arr
            r0 = i * step + 1
            c0 = j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (o5x) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
REPEAT_STEP = 2
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 300

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _precompute_masks():
    """Enumerate all valid LR-symmetric 3x3 masks, keyed by active pixel count."""
    weights = [len(g) for g in _LR_GROUPS]
    by_count = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks = []
        for k in range(1, len(_LR_GROUPS) + 1):
            for combo in combinations(range(len(_LR_GROUPS)), k):
                if sum(weights[i] for i in combo) != inactive_count:
                    continue
                mask = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
                for i in combo:
                    for r, c in _LR_GROUPS[i]:
                        mask[r, c] = False
                if mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_lr_pattern(color_a, color_b):
    """Generate a random LR-symmetric 3x3 pattern using two colors (guaranteed non-uniform)."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask():
    """Pick a random valid minority mask (uniform over active count, then uniform within count)."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _placement_candidates(mask, given_cells):
    """Find all concrete 3x3 block placements consistent with the shown minority cells."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in given_cells:
        existing[r, c] = True

    candidates = []
    for top in range(PLACEMENT_RANGE):
        for left in range(PLACEMENT_RANGE):
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[top : top + MASK_SIZE, left : left + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[top : top + MASK_SIZE, left : left + MASK_SIZE] & ~mask):
                continue
            candidates.append((top, left))
    return candidates


def _repeat_class_candidates(mask, given_cells):
    """Collapse concrete placements into the unique output-determining repeat classes."""
    placements = _placement_candidates(mask, given_cells)
    return sorted({(top % REPEAT_STEP, left) for top, left in placements})


def _choose_given_subset(mask, top, left):
    """Choose a subset of active pixels that uniquely determines one vertical repeat class."""
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    target = (top % REPEAT_STEP, left)
    for size in sizes:
        valid = []
        for subset in combinations(active, size):
            given_cells = [(top + r, left + c) for r, c in subset]
            if _repeat_class_candidates(mask, given_cells) == [target]:
                valid.append(subset)
        if valid:
            return random.choice(valid)
    return None


def _cell_origin(i, j):
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator, majority, minority, active_cells):
    """Construct a 29x29 grid: fill all cells with majority, then overwrite active_cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _repeated_active_cells(mask, parity, left):
    """Active cell coordinates after repeating the 3x3 block vertically within one repeat class."""
    cells: set[tuple[int, int]] = set()
    for top in range(parity, PLACEMENT_RANGE, REPEAT_STEP):
        for r, c in np.argwhere(mask):
            cells.add((top + int(r), left + int(c)))
    return sorted(cells)


def _gen_attempt():
    """Single generation attempt: build output, choose input subset, verify with solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    parity = random.randrange(REPEAT_STEP)
    left = random.randrange(PLACEMENT_RANGE)
    valid_tops = list(range(parity, PLACEMENT_RANGE, REPEAT_STEP))
    top = random.choice(valid_tops)

    active_cells = _repeated_active_cells(mask, parity, left)
    out = _build_grid(separator, majority, minority, active_cells)

    subset = _choose_given_subset(mask, top, left)
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


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 input")
