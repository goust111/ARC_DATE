# Combo variant of b99e7126 (index 3)
# combo_spec: D1/3+D12/0
# Name: 对称性破缺检测 + 旋转对称铺贴

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# Design analysis
# ===============
# fused_rule:
#   backbone_kept:
#     - 继续使用原题的 29x29 画布、7x7 单元格网格、3x3 单元格内部、以及“单元格内
#       3x3 图案在单元格层面再次放大”的分形补全过程。
#   D12_component:
#     - 多数背景不再是同一个固定单元格，而是来自同一个 3x3 基图案的四个旋转态。
#     - 设背景基图案为 M0，则格子 (r,c) 的期望背景是 `rot90(M0, phase + r + c)`。
#       因此向右或向下移动一格，背景都会顺时针再转 90 度。
#   D1_component:
#     - 不能再靠“所有不匹配背景的格子都属于目标”来解题；否则它会退化成 D12 单变体。
#     - 为此，输入里放两类异常格：
#         1. 真正的 fractal seed 图案 A：只显示完整 3x3 半转对称 mask 的一个真子集。
#         2. decoy 图案 D：额外的一对 180 度对置异常格，它们也违背背景旋转，但不参与分形。
#     - 于是 solver 必须先恢复旋转铺贴背景，再在多类异常格中找出“哪一类导致全局
#       180 度对置关系只坏了一对”的图案，才能识别真正种子。
#   concretization:
#     - 为了让“旋转对称破缺”在 7x7 奇数网格上清楚可见，最终的 3x3 fractal block 固定
#       居中放在 cells[2:5, 2:5]；其 mask 要求 180 度旋转对称。
#     - 输出时，完整 A-block 与 decoy 对一起构成一个关于网格中心的 180 度对置布局。
#     - 输入时，只保留 3-4 个 A-cells，且恰好缺失一个对置伙伴；decoy 仍保持成对。
#   behavior_trace:
#     - 原题：
#         1. 统计单元格图案频数。
#         2. 最少数图案就是 minority。
#         3. 用 minority 的主色 mask 对齐已有 minority 位置并补全输出。
#     - 本融合变体：
#         1. 先根据绝大多数格子恢复四相位旋转背景。
#         2. 所有“不等于该位置期望背景”的格子分成两类：A 与 D。
#         3. 只把其中一类当作候选种子；若把它标为 1、其他异常标为 2，则整张 7x7
#            类型图在 180 度对置下必须恰好只有一对不一致，而且这对不一致必须是
#            `candidate vs background`，不能是 `other-special vs background`。
#         4. 该候选的主色 mask 还必须是居中的 180 度对称 3x3 mask，并且已有 A-cells
#            是它的真子集。满足这些条件的唯一候选才是真正的 fractal seed。
#     - 具体小例子：
#         1. 背景沿行/列呈现 `0,1,2,3` 四相位旋转。
#         2. 居中的 3x3 block 对应 mask 有 7 个 active；输出里这 7 个位置都应是 A。
#         3. 输入只给出其中 4 个，其中一对对置 active 只出现左上那一个，右下伙伴缺失。
#         4. 另有一对 block 外的 decoy 异常格已经成对出现。于是“特殊格总数”并不能
#            直接告诉你谁是目标，必须看哪一类破坏了对置旋转。
#   complexity_calibration:
#     - 若强行要求 90 度对称 mask，则 3x3 mask 的 active 数只能是 1/5/9 一类，无法保留
#       原题 6-7 个 active 的密度，因此这里把 D1 的“旋转对称破缺”具体化为 180 度对置。
#       这样既保留原题分形可见性，也仍然是明确的旋转对称规则。
#     - 背景模板只从少量手工挑选的非自旋转图案中选，保证四个旋转态都清楚可辨。
#     - decoy 固定为一对，不引入更多类别，既能阻止退化回 D12-only，又不会把画面搞乱。
#   implicit_assumption_audit:
#     - 原题的 `minority_pat = min(counts...)` 失效：这里最稀有图案可能是 decoy，而不是目标。
#     - D12-only 的“所有背景不匹配格都属于同一个 anomaly pattern”也失效：这里故意有 A 与 D
#       两类异常，因此不能直接把所有 mismatch 视作同一图案。
#     - 原题默认 3x3 block 可在任意偏移放置；这里 D1 旋转对称要求 block 居中。
#     - 原题把 block 内 inactive 位置写回单一 majority pattern；这里必须写回该位置的期望
#       旋转背景态。
#   legality:
#     - 居中 3x3 block 永远合法，范围固定为 rows/cols 2..4。
#     - decoy 只放在 block 外，并以 180 度对置成对出现；因此不会越界，也不会与 block 重叠。
#     - 输入中的给定 A-cells 只取自 mask 的 active 位置，且严格少于完整 active 数。
#   visual_observability:
#     - 人眼先能看出背景 tile 在逐格旋转，再能看出有几格完全没跟上这个旋转规律。
#     - 这些异常格里，只有一种图案导致“对面少了一个”，这正是 D1 的破缺线索。
#     - 输出则表现为：中心 3x3 单元块被补成完整的半转对称 fractal，而 decoy 对保持不变。
#   cleanliness:
#     - 全图只多出一个 decoy 类别，总颜色数控制在 5。
#     - 特殊格只有两类且都还是标准 3x3 单元格，无额外标记层或局部噪声。
#   solver_generator_contract:
#     - 背景四个旋转态必须都来自同一 majority orbit，且 phase 可唯一恢复。
#     - 输入中 mismatch pattern 恰有两类：真正 minority A 与 decoy D。
#     - A 的主色 mask 是 3x3、180 度对称、active 数为 6 或 7、且覆盖四条边。
#     - A 的给定子集在 180 度对置下恰好只坏一对；decoy 对在同一对置规则下保持完整。


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 500

_HT_GROUPS = [
    ((0, 0), (2, 2)),
    ((0, 2), (2, 0)),
    ((0, 1), (2, 1)),
    ((1, 0), (1, 2)),
    ((1, 1),),
]

_MAJORITY_MASKS = [
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=bool),
]

_DECOY_MASKS = [
    np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool),
]


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
    return np.rot90(arr, -int(k) % 4)


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(np.asarray(arr, dtype=int).ravel().tolist())


def _canonical_rotation_key(arr: np.ndarray) -> tuple[int, ...]:
    return min(_pattern_key(_rot_cw(arr, k)) for k in range(4))


def _is_half_turn_symmetric(arr: np.ndarray) -> bool:
    return np.array_equal(arr, arr[::-1, ::-1])


def _is_vertical_symmetric(arr: np.ndarray) -> bool:
    return np.array_equal(arr, arr[:, ::-1])


def _is_horizontal_symmetric(arr: np.ndarray) -> bool:
    return np.array_equal(arr, arr[::-1, :])


def _covers_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _centered_anchor(n: int, mh: int, mw: int) -> tuple[int, int]:
    assert (n - mh) % 2 == 0 and (n - mw) % 2 == 0, "mask must center in the cell lattice"
    return (n - mh) // 2, (n - mw) // 2


def _half_turn_pos(pos: tuple[int, int], n: int) -> tuple[int, int]:
    r, c = pos
    return n - 1 - r, n - 1 - c


def _rot180_mismatch_pairs(type_grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    n = type_grid.shape[0]
    mismatches: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i in range(n):
        for j in range(n):
            a = (i, j)
            b = _half_turn_pos(a, n)
            if a >= b:
                continue
            if int(type_grid[a]) != int(type_grid[b]):
                mismatches.append((a, b))
    return mismatches


def _centered_mask_fits(mask: np.ndarray, existing: np.ndarray) -> bool:
    n = existing.shape[0]
    mh, mw = mask.shape
    top, left = _centered_anchor(n, mh, mw)
    block = np.zeros_like(existing)
    block[top : top + mh, left : left + mw] = True
    if np.any(existing & ~block):
        return False
    ex_block = existing[top : top + mh, left : left + mw]
    if np.any(ex_block & ~mask):
        return False
    return True


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _expected_majority(majority_base: np.ndarray, phase: int, i: int, j: int) -> np.ndarray:
    return _rot_cw(majority_base, phase + i + j)


def _extract_cells(g: np.ndarray, step: int) -> tuple[list[list[np.ndarray]], int]:
    n = (g.shape[0] - 1) // step
    inner = step - 1
    cell_arr: list[list[np.ndarray]] = []
    for i in range(n):
        row: list[np.ndarray] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            row.append(g[r0 : r0 + inner, c0 : c0 + inner].copy())
        cell_arr.append(row)
    return cell_arr, inner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the centered fractal block by finding which anomaly pattern uniquely breaks a rotational pairing rule.

    Input structure:
    - The grid is split into 7x7 cells with 3x3 interiors separated by single-color divider lines.
    - Most cells belong to one majority motif whose expected appearance depends on position: moving right or down rotates that motif by 90 degrees.
    - Cells that do not match the recovered rotating background come in two patterns: a few true seed cells inside a centered 3x3 block, plus one extra decoy pair placed outside that block.
    - The true seed cells are arranged as an incomplete 180-degree-symmetric subset of a larger centered fractal mask, so exactly one opposite partner is missing in the input.

    Rule:
    - First reconstruct the rotating majority background orbit and its global phase.
    - Then examine each anomaly pattern separately. The correct seed is the only one whose dominant-color mask is a centered 3x3 half-turn-symmetric blueprint and whose positions cause exactly one broken 180-degree opposite pair against the rest of the lattice.
    - Use that seed mask at cell scale: active mask positions become the seed pattern, inactive positions inside the centered block revert to their expected rotated majority tiles.
    - Leave the divider lines and the decoy anomaly pair unchanged.

    Key insight: not every background mismatch belongs to the fractal. The solver has to combine two inferences: recover the quarter-turn tiled background, then identify which anomaly type is the unique symmetry breaker in the centered 180-degree layout.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    cell_arr, inner = _extract_cells(g, step)

    orbit_counts: dict[tuple[int, ...], int] = {}
    orbit_examples: dict[tuple[int, ...], np.ndarray] = {}
    for i in range(n):
        for j in range(n):
            arr = cell_arr[i][j]
            orbit = _canonical_rotation_key(arr)
            orbit_counts[orbit] = orbit_counts.get(orbit, 0) + 1
            if orbit not in orbit_examples:
                orbit_examples[orbit] = arr

    assert len(orbit_counts) >= 2, "need a majority rotation orbit plus anomaly patterns"
    majority_orbit = max(orbit_counts.items(), key=lambda kv: kv[1])[0]
    majority_anchor = orbit_examples[majority_orbit]

    phase_scores: list[tuple[int, int]] = []
    for phase in range(4):
        matches = 0
        for i in range(n):
            for j in range(n):
                expected = _expected_majority(majority_anchor, phase, i, j)
                if np.array_equal(cell_arr[i][j], expected):
                    matches += 1
        phase_scores.append((matches, phase))

    best_match_count = max(score for score, _ in phase_scores)
    best_phases = [phase for score, phase in phase_scores if score == best_match_count]
    assert len(best_phases) == 1, "majority rotation phase must be uniquely determined"
    phase = best_phases[0]

    expected_cells: list[list[np.ndarray]] = []
    mismatch_by_pattern: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    pattern_samples: dict[tuple[int, ...], np.ndarray] = {}
    mismatch_positions: list[tuple[int, int]] = []
    for i in range(n):
        expected_row: list[np.ndarray] = []
        for j in range(n):
            expected = _expected_majority(majority_anchor, phase, i, j)
            expected_row.append(expected)
            if not np.array_equal(cell_arr[i][j], expected):
                pos = (i, j)
                key = _pattern_key(cell_arr[i][j])
                mismatch_by_pattern.setdefault(key, []).append(pos)
                pattern_samples[key] = cell_arr[i][j]
                mismatch_positions.append(pos)
        expected_cells.append(expected_row)

    assert len(mismatch_by_pattern) >= 2, "fused variant expects at least seed and decoy anomaly patterns"

    solutions: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for candidate_key, candidate_positions in mismatch_by_pattern.items():
        candidate_arr = pattern_samples[candidate_key]
        vals, cnts = np.unique(candidate_arr, return_counts=True)
        active_color = int(vals[cnts.argmax()])
        mask = candidate_arr == active_color
        if int(mask.sum()) not in ACTIVE_PIXEL_COUNTS:
            continue
        if not _is_half_turn_symmetric(mask):
            continue
        if not _covers_all_edges(mask):
            continue

        existing = np.zeros((n, n), dtype=bool)
        for pos in candidate_positions:
            existing[pos] = True
        if int(existing.sum()) >= int(mask.sum()):
            continue
        if not _centered_mask_fits(mask, existing):
            continue

        type_grid = np.zeros((n, n), dtype=int)
        for pos in mismatch_positions:
            type_grid[pos] = 2
        for pos in candidate_positions:
            type_grid[pos] = 1

        broken_pairs = _rot180_mismatch_pairs(type_grid)
        if len(broken_pairs) != 1:
            continue
        a, b = broken_pairs[0]
        labels = {int(type_grid[a]), int(type_grid[b])}
        if labels != {0, 1}:
            continue

        top, left = _centered_anchor(n, *mask.shape)
        solutions.append((candidate_arr, mask, top, left))

    assert len(solutions) == 1, "exactly one anomaly pattern should satisfy the fused symmetry-break rule"
    minority_arr, mask, top, left = solutions[0]

    out = g.copy()
    mh, mw = mask.shape
    for i in range(mh):
        for j in range(mw):
            target = minority_arr if mask[i, j] else expected_cells[top + i][left + j]
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target
    return out


def _precompute_minority_masks() -> dict[int, list[np.ndarray]]:
    weights = [len(group) for group in _HT_GROUPS]
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
        for k in range(1, len(_HT_GROUPS) + 1):
            for combo in combinations(range(len(_HT_GROUPS)), k):
                if sum(weights[idx] for idx in combo) != inactive_count:
                    continue
                mask = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
                for idx in combo:
                    for r, c in _HT_GROUPS[idx]:
                        mask[r, c] = False
                if not _covers_all_edges(mask):
                    continue
                if _is_vertical_symmetric(mask.astype(int)) or _is_horizontal_symmetric(mask.astype(int)):
                    continue
                masks.append(mask)
        by_count[active_count] = masks
    return by_count


def _validate_majority_masks() -> None:
    for mask in _MAJORITY_MASKS:
        keys = {_pattern_key(_rot_cw(mask.astype(int), k)) for k in range(4)}
        assert len(keys) == 4, "majority templates must expose four distinct rotations"


def _validate_decoy_masks() -> None:
    for mask in _DECOY_MASKS:
        assert not _is_half_turn_symmetric(mask.astype(int)), "decoy templates should not mimic the seed symmetry"


_MINORITY_MASKS_BY_COUNT = _precompute_minority_masks()
_validate_majority_masks()
_validate_decoy_masks()


def _random_majority_pattern(color_a: int, color_b: int) -> np.ndarray:
    mask = random.choice(_MAJORITY_MASKS)
    return np.where(mask, color_b, color_a).astype(int)


def _random_minority_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MINORITY_MASKS_BY_COUNT[active_count])


def _random_decoy_pattern(color_a: int, color_b: int) -> np.ndarray:
    mask = random.choice(_DECOY_MASKS)
    return np.where(mask, color_b, color_a).astype(int)


def _build_grid(
    separator: int,
    majority_base: np.ndarray,
    phase: int,
    minority: np.ndarray,
    decoy: np.ndarray,
    active_cells: list[tuple[int, int]],
    decoy_cells: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, _expected_majority(majority_base, phase, i, j), *_cell_origin(i, j))
    for i, j in decoy_cells:
        paste(grid, decoy, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _choose_decoy_pair(block_cells: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    candidates: list[list[tuple[int, int]]] = []
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            a = (i, j)
            b = _half_turn_pos(a, CELL_GRID_N)
            if a >= b or a == b:
                continue
            if a in block_cells or b in block_cells:
                continue
            candidates.append([a, b])
    if not candidates:
        return None
    return random.choice(candidates)


def _subset_break_count(subset: tuple[tuple[int, int], ...]) -> int:
    shown = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    for r, c in subset:
        shown[r, c] = True

    breaks = 0
    for i in range(MASK_SIZE):
        for j in range(MASK_SIZE):
            a = (i, j)
            b = (MASK_SIZE - 1 - i, MASK_SIZE - 1 - j)
            if a >= b:
                continue
            if shown[a] != shown[b]:
                breaks += 1
    return breaks


def _arrangement_break_count(given_cells: list[tuple[int, int]], decoy_cells: list[tuple[int, int]]) -> tuple[int, set[int]]:
    types = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=int)
    for pos in decoy_cells:
        types[pos] = 2
    for pos in given_cells:
        assert types[pos] == 0, "seed / decoy overlap"
        types[pos] = 1
    broken = _rot180_mismatch_pairs(types)
    labels: set[int] = set()
    if broken:
        a, b = broken[0]
        labels = {int(types[a]), int(types[b])}
    return len(broken), labels


def _choose_given_subset(mask: np.ndarray, phase: int, top: int, left: int, decoy_cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...] | None:
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    diversified: list[tuple[tuple[int, int], ...]] = []
    fallback: list[tuple[tuple[int, int], ...]] = []
    for size in sizes:
        for subset in combinations(active, size):
            if _subset_break_count(subset) != 1:
                continue
            given_cells = [(top + r, left + c) for r, c in subset]
            break_count, labels = _arrangement_break_count(given_cells, decoy_cells)
            if break_count != 1 or labels != {0, 1}:
                continue
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


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, min_exclusive, decoy_exclusive = random.sample(range(COLOR_POOL), 5)
    majority = _random_majority_pattern(shared, maj_exclusive)
    minority_mask = _random_minority_mask()
    minority = np.where(minority_mask, min_exclusive, shared).astype(int)
    decoy = _random_decoy_pattern(shared, decoy_exclusive)

    if np.array_equal(minority, decoy):
        return None
    if any(np.array_equal(minority, _rot_cw(majority, k)) for k in range(4)):
        return None
    if any(np.array_equal(decoy, _rot_cw(majority, k)) for k in range(4)):
        return None

    phase = random.randrange(4)
    top, left = _centered_anchor(CELL_GRID_N, MASK_SIZE, MASK_SIZE)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(minority_mask)]
    block_cells = {(top + r, left + c) for r in range(MASK_SIZE) for c in range(MASK_SIZE)}

    decoy_cells = _choose_decoy_pair(block_cells)
    if decoy_cells is None:
        return None

    out = _build_grid(separator, majority, phase, minority, decoy, active_cells, decoy_cells)

    subset = _choose_given_subset(minority_mask, phase, top, left, decoy_cells)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, phase, minority, decoy, given_cells, decoy_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
