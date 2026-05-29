# Variant of b99e7126
# Dimension: D12 — 背景单元格图案的对称性要求 (meta)
# Change: 多数单元格使用完全相同的固定图案（简单重复铺贴）... → 多数单元格图案具有旋转对称：相邻单元格旋转90度

from functools import reduce
from math import gcd

import numpy as np
import random
from itertools import combinations
from arc_utils import paste

# Design analysis
# ---
# concrete_behavior:
#   grid:
#     - 保持原题的 29x29 像素网格、7x7 单元格、每格 3x3 内部、1 像素分隔线。
#     - 仍然存在一个 3x3 的“异常图案”，它的主色像素在单元格级别上定义一个 3x3 mask。
#   new_majority_rule:
#     - 原题多数格是一个固定 3x3 图案的平铺；这里改成一个“旋转轨道”。
#     - 选一个明显不具 90/180 度自对称的多数基图案 `M0`。
#     - 对任意格子 (r, c)，其期望背景图案为 `cw_rotate(M0, phase + r + c)`。
#     - 因此向右走一格或向下走一格，图案都顺时针再转 90 度。
#   anomaly_behavior:
#     - 输入中的 3-4 个异常格都使用同一个固定异常图案 `A`，而不是按背景规则旋转。
#     - 输出与原题一样，是把 `A` 内部主色形成的 mask 按单元格尺度放大后补全。
#     - 只是“非 active 的位置”不能再写回一个固定多数图案，而必须写回该位置应有的旋转态。
#   trace_example:
#     - 原题：若多数格全是同一个 `M`，solver 只需找出少数格 `A`，然后在 block 内把 False 位置写回 `M`。
#     - 本变体：假设 `(2, 3)` 处应为 `cw_rotate(M0, 1)`，但输入里放了异常格 `A`。
#     - solver 先恢复整个 7x7 背景旋转相位，再识别所有“不等于该位置期望旋转态”的格子为异常格。
#     - 最后补全 active 位置为 `A`，inactive 位置写回各自的期望旋转态，而不是单一常量图案。
# complexity_calibration:
#   - 若多数基图案本身可旋转后不变，人眼看不出 D12，solver 也无法恢复相位；因此生成器只用显著不对称模板。
#   - 若多数模板太随机，视觉上会乱；因此只从少量手工设计的 L/钩形模板中抽样，既能看出旋转，又不会噪声过高。
#   - 若输入显示的异常格都落在同一旋转相位上，D12 线索会弱；因此生成器优先选择覆盖至少 2 个背景相位的 given subset。
# old_main_assumption_audit:
#   - `majority_pat = max(counts.items(), ...)` 依赖“多数格字面完全相同”；变体后不成立，因为多数格分成 4 个旋转态。
#   - `existing[i,j] = cell_pat[i][j] == minority_pat` 依赖“异常定义为字面少数图案”；变体后异常应定义为“不满足该位置旋转关系”。
#   - `target_pat = ... else patterns[majority_pat]` 依赖“所有 inactive 位置都写同一个多数图案”；变体后必须按坐标恢复不同旋转态。
#   - 原代码不需要相位推断；变体后若不先恢复相位，异常格和正常旋转格无法区分。
# placement_legality:
#   - mask 仍固定为 3x3，因此放置偏移仍在 0..4。
#   - 生成器仍采用 output-first：先生成完整输出，再挑选 3-4 个 active 格保留到输入。
#   - 输入中的异常格都位于最终 3x3 block 内，block 外全部保持合法背景旋转态，不重叠、不越界。
# visual_observability:
#   - 人眼能先看到背景图案在行列上持续旋转，再看到少数格没有跟着旋转且内部颜色也不同。
#   - 输出中异常格形成的 3x3 单元级图案，仍保留原题的“分形放大”可见性。
# visual_cleanliness:
#   - 全局只用 4 种颜色，与原题一致；背景仍只用两色，异常也只用两色，分隔线单色。
#   - 多数模板数量受控，避免每个格子都像独立随机噪声。
# color_inference:
#   - 不依赖颜色 id 排序，也不根据颜色编号决定方向。
#   - 异常 mask 的 active 颜色仍由频数支撑：6-7 个 active 像素 > 2-3 个 inactive 像素，肉眼可见。
# new_main_assumption_review:
#   - solver 假设多数背景全部来自同一旋转轨道，且 4 个旋转态彼此可区分；生成器通过模板筛选保证。
#   - solver 假设所有异常格字面相同；生成器始终只放一个固定异常图案。
#   - solver 假设给定异常格是最终 active 集合的真子集，且 placement 唯一；生成器沿用原题的唯一性检查并自验证 `main(inp)==out`。


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


def _rot_cw(arr: np.ndarray, k: int) -> np.ndarray:
    """Rotate a 2D pattern clockwise by k quarter-turns."""
    return np.rot90(arr, -int(k) % 4)


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    """Flatten a pattern into a hashable tuple."""
    return tuple(np.asarray(arr, dtype=int).ravel().tolist())


def _canonical_rotation_key(arr: np.ndarray) -> tuple[int, ...]:
    """Canonical key for a rotation orbit: min key among all 4 clockwise rotations."""
    return min(_pattern_key(_rot_cw(arr, k)) for k in range(4))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the fractal anomaly overlay on top of a rotationally tiled background.

    Input structure:
    - The grid is split into equal 3x3 cell interiors by 1-pixel separator lines.
    - Most cells belong to one asymmetric majority motif, but neighboring cells are quarter-turn rotations of it rather than exact copies.
    - A few cells share one fixed anomaly pattern; those are the cells that break the rotation cycle.

    Rule:
    - Reconstruct the majority rotation phase over the whole cell lattice.
    - Use the anomaly cell's dominant-color mask as a 3x3 blueprint at cell scale.
    - Find the unique placement consistent with the given anomaly cells, then fill every active mask position with the anomaly pattern and every inactive position with that cell's expected rotated majority motif.

    Key insight: The exceptional cells are defined relative to a rotating background, so the solver must infer the quarter-turn tiling first and only then apply the original fractal completion idea.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    cell_arr: list[list[np.ndarray]] = []
    orbit_counts: dict[tuple[int, ...], int] = {}
    orbit_examples: dict[tuple[int, ...], np.ndarray] = {}
    for i in range(n):
        row: list[np.ndarray] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            row.append(arr)
            orbit = _canonical_rotation_key(arr)
            orbit_counts[orbit] = orbit_counts.get(orbit, 0) + 1
            if orbit not in orbit_examples:
                orbit_examples[orbit] = arr
        cell_arr.append(row)

    assert len(orbit_counts) >= 2, "need majority orbit plus anomaly pattern"
    majority_orbit = max(orbit_counts.items(), key=lambda kv: kv[1])[0]
    majority_anchor = orbit_examples[majority_orbit]

    phase_scores: list[tuple[int, int]] = []
    for phase in range(4):
        matches = 0
        for i in range(n):
            for j in range(n):
                expected = _rot_cw(majority_anchor, phase + i + j)
                if np.array_equal(cell_arr[i][j], expected):
                    matches += 1
        phase_scores.append((matches, phase))

    best_matches = max(score for score, _ in phase_scores)
    best_phases = [phase for score, phase in phase_scores if score == best_matches]
    assert len(best_phases) == 1, "rotation phase must be uniquely determined"
    phase = best_phases[0]

    expected_cells: list[list[np.ndarray]] = []
    mismatch_pos: list[tuple[int, int]] = []
    mismatch_patterns: dict[tuple[int, ...], np.ndarray] = {}
    for i in range(n):
        row: list[np.ndarray] = []
        for j in range(n):
            expected = _rot_cw(majority_anchor, phase + i + j)
            row.append(expected)
            if not np.array_equal(cell_arr[i][j], expected):
                mismatch_pos.append((i, j))
                mismatch_patterns[_pattern_key(cell_arr[i][j])] = cell_arr[i][j]
        expected_cells.append(row)

    assert mismatch_pos, "need at least one anomaly cell"
    assert len(mismatch_patterns) == 1, "all anomaly cells must share the same pattern"
    minority_arr = next(iter(mismatch_patterns.values()))

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = (minority_arr == active_color)
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i, j in mismatch_pos:
        existing[i, j] = True

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
    assert len(candidates) == 1, "mask placement must be uniquely determined"
    top, left = candidates[0]

    out = g.copy()
    for i in range(n):
        for j in range(n):
            inside = top <= i < top + mh and left <= j < left + mw
            if inside and mask[i - top, j - left]:
                target = minority_arr
            else:
                target = expected_cells[i][j]
            r0 = i * step + 1
            c0 = j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target
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

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]

_MAJORITY_MASKS = [
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=bool),
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


def _validate_majority_masks():
    """Fail fast if a template collapses under rotation and would hide D12."""
    for mask in _MAJORITY_MASKS:
        keys = {_pattern_key(_rot_cw(mask.astype(int), k)) for k in range(4)}
        assert len(keys) == 4, "majority templates must have four distinct rotations"


_MASKS_BY_COUNT = _precompute_masks()
_validate_majority_masks()


def _random_majority_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Pick a visually clean asymmetric template and color it with two colors."""
    mask = random.choice(_MAJORITY_MASKS)
    return np.where(mask, color_b, color_a).astype(int)


def _random_minority_mask() -> np.ndarray:
    """Pick a random valid minority mask (uniform over active count, then within count)."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _placement_candidates(mask, subset, top, left):
    """Find all (t, l) offsets where the given subset of active cells is consistent with mask."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True

    candidates = []
    for t in range(PLACEMENT_RANGE):
        for l in range(PLACEMENT_RANGE):
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[t : t + MASK_SIZE, l : l + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[t : t + MASK_SIZE, l : l + MASK_SIZE] & ~mask):
                continue
            candidates.append((t, l))
    return candidates


def _choose_given_subset(mask, top, left, phase):
    """Choose a unique subset of active cells, preferring multiple background rotation phases."""
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    diversified = []
    fallback = []
    for size in sizes:
        valid = [
            subset
            for subset in combinations(active, size)
            if _placement_candidates(mask, subset, top, left) == [(top, left)]
        ]
        if not valid:
            continue
        for subset in valid:
            orientations = {(phase + top + r + left + c) % 4 for r, c in subset}
            if len(orientations) >= 2:
                diversified.append(subset)
            else:
                fallback.append(subset)
        if diversified:
            return random.choice(diversified)
        if fallback:
            return random.choice(fallback)
    return None


def _cell_origin(i, j):
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _expected_majority(majority_base: np.ndarray, phase: int, i: int, j: int) -> np.ndarray:
    """Expected majority pattern at cell (i, j) under the global rotation phase."""
    return _rot_cw(majority_base, phase + i + j)


def _build_grid(separator, majority_base, phase, minority, active_cells):
    """Construct a full grid with rotating majority background, then overwrite active cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, _expected_majority(majority_base, phase, i, j), *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _gen_attempt():
    """Single generation attempt: build output, choose input subset, verify with solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_majority_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    phase = random.randrange(4)
    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    out = _build_grid(separator, majority, phase, minority, active_cells)

    subset = _choose_given_subset(mask, top, left, phase)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, phase, minority, given_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 D12 input")
