# Variant of b99e7126
# Dimension: D5 — 变换：叠加时的覆盖规则 (transform)
# Change: 放大图案中非背景色的单元格覆写原始网格中对应位置的单元格内容... → 擦除式叠加：放大图案的活跃区域变为背景色

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random
from arc_utils import paste

# Design analysis
# variant_spec:
#   source_task_baseline:
#     reference_task: 58f5dbd5
#     borrowed_mechanism: "前景图案不被贴上去，而是把目标区域挖成背景色"
#     complexity_match: "只改变覆盖结果，不改变对象发现、定位和放大机制"
#   concretization:
#     scene:
#       - "29x29 网格仍由 7x7 个 3x3 单元格组成，单元格之间用 1 像素分隔线隔开。"
#       - "输入里大多数单元格仍是同一个 majority 纹理；少数几个 clue 单元格仍是 minority 纹理。"
#       - "minority 纹理内部的 3x3 像素图案仍是放大蓝图，其高频色定义活跃像素。"
#       - "输出不再把蓝图活跃位置写成 minority 单元格，而是把这些整块单元格内部擦成分隔线颜色，形成 3x3 的空洞。"
#     traced_example:
#       - "若 minority 掩码右上角像素为活跃，原题会把对应的大单元格改写成 minority 纹理。"
#       - "本变体在同一位置改写成全分隔线色的 3x3 实心块，原先已有的 minority clue 也会一起被抹掉。"
#       - "差异只发生在最后覆盖阶段：定位逻辑不变，落笔颜色从 minority 改为 background。"
#   complexity_calibration:
#     - "保持原题的 3x3 掩码、6-7 个活跃像素、3-4 个已给 clue，避免定位过易或过难。"
#     - "擦除色固定为全局可见的分隔线颜色，视觉差异稳定且不依赖颜色编号。"
#     - "输入仍保留 minority clue，因此求解仍需先识别异常单元格，再理解其内部图案。"
#   implicit_assumption_audit:
#     - "原 main() 的 `_infer_cell_grid_params` 只依赖整行整列分隔线；擦除发生在单元格内部，不破坏此假设。"
#     - "原 `counts`/`minority_pat` 推断依赖输入里只有 majority 与 minority 两类单元格；生成器继续维持这一点。"
#     - "原 `active_color = argmax(counts)` 依赖 minority 的前景色是多数色；生成器继续令活跃像素数为 6 或 7。"
#     - "原最终循环把活跃位置写成 minority，这一旧前提在变体中失效，必须改成写入统一的擦除单元格。"
#     - "若无条件把擦除块也用于输入，会破坏 minority clue 的存在并导致无法恢复掩码；因此擦除只出现在输出。"
#   placement_legality:
#     - "3x3 掩码仍放在 7x7 单元格网格中的某个 0..4 偏移处，不越界。"
#     - "输入只在掩码活跃位置中保留 3 或 4 个 minority clue；其余位置保持 majority，保证唯一定位。"
#     - "输出把全部活跃位置擦除为分隔线色单元格，不会与分隔线之外的对象重叠，因为作用域仍是单元格内部。"
#   visual_observability:
#     - "从样例上看，输入中的异常纹理像是在提示一个更大的形状；输出则在该形状处出现整块空洞。"
#     - "由于空洞颜色与分隔线一致，解题者能自然把它理解为被挖空的背景，而不是另一种新纹理。"
#   cleanliness:
#     - "每个单元格内部仍只有一种清晰纹理：majority、minority clue 或纯背景洞块；不会出现多层局部叠加。"
#   color_inference:
#     - "不使用颜色 id 排序；背景色直接由可见分隔线提供。"
#     - "颜色角色仍限制为 4 种：分隔线、majority 独占色、shared 色、minority 独占色。"
#   post_change_solver_assumptions:
#     - "main() 现在假设要擦除到的背景色就是分隔线色 `grid[0,0]`；生成器用全分隔线色洞块满足此假设。"
#     - "main() 仍假设 inactive 位置应恢复为 majority；生成器始终以 majority 作为输入基底来保证这一点。"


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the minority cell's inner motif as a cell-grid stencil, then carve that stencil out of the grid.

    Input structure:
    - The canvas is partitioned into equal 3x3 cell interiors by uniform divider lines.
    - Nearly every cell shares one majority interior pattern, while a small clue subset shows one minority pattern.

    Rule:
    - Read the minority cell's 3x3 interior as a binary mask, with its dominant color marking active pixels.
    - Place that mask on the larger 7x7 cell lattice so every clue minority cell lands on an active mask position.
    - For every active position in the placed mask, erase the corresponding whole cell interior to the divider color instead of stamping in the minority pattern.

    Key insight: The minority cell still encodes a fractal blueprint, but in this variant the blueprint is used as a carving stencil that punches background-colored holes into the larger grid.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1
    bg = int(g[0, 0])

    # Extract interior patterns per cell.
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

    # Find minority/majority cell patterns from the input clues.
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            p = cell_pat[i][j]
            counts[p] = counts.get(p, 0) + 1
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns (minority clues + majority)"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

    # Active pixels are the dominant color inside the minority clue pattern.
    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat

    # Find the unique placement where the observed minority clues occupy active mask positions.
    candidates: list[tuple[int, int]] = []
    for top in range(0, n - mh + 1):
        for left in range(0, n - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "must have a unique mask placement aligning existing minority clues"
    top, left = candidates[0]

    erased_cell = np.full((inner, inner), bg, dtype=int)
    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target_pat = erased_cell if mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (o5x) ---------------

# --- CFG: tunable generation parameters ---
CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 300

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
    """Enumerate all valid LR-symmetric 3x3 masks, keyed by active pixel count."""
    weights = [len(group) for group in _LR_GROUPS]
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
        for k in range(1, len(_LR_GROUPS) + 1):
            for combo in combinations(range(len(_LR_GROUPS)), k):
                if sum(weights[idx] for idx in combo) != inactive_count:
                    continue
                mask = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
                for idx in combo:
                    for r, c in _LR_GROUPS[idx]:
                        mask[r, c] = False
                if mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a random LR-symmetric 3x3 pattern using two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(value == choices[0] for value in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask() -> np.ndarray:
    """Pick a random valid minority mask."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _placement_candidates(mask: np.ndarray, subset, top: int, left: int) -> list[tuple[int, int]]:
    """Find all offsets where the subset of clue cells is consistent with the mask."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True

    candidates: list[tuple[int, int]] = []
    for cand_top in range(PLACEMENT_RANGE):
        for cand_left in range(PLACEMENT_RANGE):
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[cand_top : cand_top + MASK_SIZE, cand_left : cand_left + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[cand_top : cand_top + MASK_SIZE, cand_left : cand_left + MASK_SIZE] & ~mask):
                continue
            candidates.append((cand_top, cand_left))
    return candidates


def _choose_given_subset(mask: np.ndarray, top: int, left: int):
    """Choose a clue subset that uniquely determines the placement."""
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(active, size)
            if _placement_candidates(mask, subset, top, left) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, minority_cells, minority: np.ndarray) -> np.ndarray:
    """Construct a 29x29 grid from majority base cells plus selected minority clue cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in minority_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _build_output(separator: int, majority: np.ndarray, mask: np.ndarray, top: int, left: int) -> np.ndarray:
    """Construct the erased output: active mask cells become solid separator-colored holes."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    erased = np.full((CELL_INNER, CELL_INNER), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for r, c in np.argwhere(mask):
        paste(grid, erased, *_cell_origin(top + int(r), left + int(c)))
    return grid


def _gen_attempt():
    """Single generation attempt: build target output, synthesize input clues, and verify."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    out = _build_output(separator, majority, mask, top, left)

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, given_cells, minority)

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
    raise RuntimeError("failed to generate a valid b99e7126-D5 input")
