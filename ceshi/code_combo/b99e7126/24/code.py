# Combo variant of b99e7126 (index 24)
# combo_spec: D1/3+D6/2
# Name: 对称性破缺检测 + 渐变铺贴

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# Design analysis
# ===============
# variant_spec:
#   kept_backbone:
#     - 保留原题的 29x29 画布、7x7 单元格网格、3x3 单元格内部，以及“把某个 3x3
#       单元格内部图案放大成 3x3 个单元格块”的分形补全过程。
#   fusion_concretization:
#     - D1/3 的核心是“异常单元格不再靠频数找，而是靠它打破了全局对称性”。
#     - D6/2 的核心是“平铺不再完全相同，而是沿一个方向做规律渐变”。
#     - 这里把两者合并成同一条推理链：先选一个对称轴（vertical / horizontal），再让
#       渐变方向就沿着这条轴对应的行或列展开，使用固定旋转相位序列 [0,1,2,3,2,1,0]。
#     - majority 与 minority 都不是单一图案，而是各自有一个 3x3 基础图案；网格中每个
#       单元格显示的是“基础图案按本行/本列相位旋转后的版本”。
#     - minority 的基础图案仍然用其主色形成 3x3 布尔 mask；这个 mask 被放大成一个
#       轴居中的 3x3 单元格块。完整输出中，该块关于所选轴镜像对称。
#     - 输入只给出 3-4 个 active 位置，其中恰好有 1 个偏轴 active 没有出现它的镜像搭档，
#       因而整张图仅在这一对镜像位置上破坏对称。
#   behavior_trace:
#     - 原逻辑：
#         1. 全网格绝大多数单元格完全相同。
#         2. 统计频数，找唯一少数图案 minority。
#         3. 读出 minority 内部主色 mask，把它当成单元格级别的蓝图完成输出。
#     - 融合后：
#         1. 先比较镜像配对的单元格，找到唯一出现 1 个不匹配对的对称轴。
#         2. 注意整张图还带有旋转渐变：例如竖直轴时，各列相位依次是 0,1,2,3,2,1,0，
#            所以同一基础图案会在不同列呈现不同朝向。
#         3. 那个不匹配对里的两格处在同一相位；把两边候选图案都“反旋转”回基础朝向后，
#            只有真正的 minority 候选能解释所有同族单元格，并且其主色 mask 能唯一落成
#            一个轴居中的 3x3 块。
#         4. 再按每个目标单元格自身的相位，把基础 majority/minority 图案旋转回去，补齐输出。
#     - 具体小例子（竖直轴）：
#         1. 7 列相位是 [0,1,2,3,2,1,0]，所以从左到中间图案逐列旋转 90 度，到右边再反向回去。
#         2. minority 分形块固定以第 3 列为镜像中心，例如 top=1, left=2。
#         3. 输入里显示了 (1,3)、(2,2)、(2,3) 三个 active，但 (2,4) 缺失。
#         4. 因为 (2,2) 与 (2,4) 相位相同，它们本该是完全相同的旋转版 minority；
#            缺的那一边仍是同相位 majority，于是出现唯一对称性破缺。
#   complexity_calibration:
#     - 只保留两种对称轴与一种渐变机制（90 度旋转），避免规则档位过多。
#     - 仍保留原题的 3x3 mask、6-7 个 active 像素、3-4 个已给 active 单元格。
#     - 渐变相位固定为对称的 7 项序列，人眼能在少量样例中直接看到“整列/整行在转”。
#     - minority 与 majority 的四个旋转态都要求互异，避免渐变退化成“其实只有两种朝向”。
#   implicit_assumption_audit:
#     - 原 main() 的 `minority_pat = min(counts...)` 依赖“异常图案出现次数最少”；
#       融合后不成立，因为 majority/minority 都会因旋转产生多个局部外观，频数不再对应角色。
#     - 原 main() 把已有 minority 单元格视为“完全相同的若干格”；融合后必须先按相位做
#       逆旋转归一化，再判断是否属于同一个基础图案家族。
#     - 原 main() 默认 3x3 块可在任意 top/left 放置；D1/3 使得块必须沿目标对称轴居中，
#       所以求解时要固定一个坐标，只在另一维搜索。
#     - 原 main() 默认网格里只有 majority/minority 两类像素外观；融合后仍只有两类基础族，
#       但每类各自展开成 4 个旋转态，因此 solver 必须验证“所有格子都能被这两族解释”。
#   legality:
#     - 轴居中的 3x3 块在 7x7 中始终合法：vertical 时 left=2，horizontal 时 top=2。
#     - mask 只从对应轴对称、另一轴不对称、且四个旋转态都不同的候选里挑选。
#     - 生成时先构造完整输出，再删掉一部分 active，因而不会出现重叠、越界或遮盖冲突。
#   visual_observability:
#     - 视觉上同时能看见两层结构：大尺度上整行/整列逐步旋转，小尺度上只有一对镜像位置
#       不一致，提示“先理解渐变，再找破缺点”。
#     - 由于镜像位置共享相位，解题者不需要借助颜色编号做任何隐藏排序，只需对比形状和朝向。
#   cleanliness:
#     - 仍然只有 divider、majority 家族、minority 家族三层，没有额外标记物或覆盖层。
#     - active 缺失只发生在一个镜像对内，其余线索要么成对出现，要么落在镜像轴上自对称。
#   color_inference:
#     - 颜色只区分 divider / majority-family / minority-family 的内部构成，不承担数值运算意义。
#     - minority 主色始终是其 3x3 基础图案中出现 6-7 次的颜色，便于从图案本身读出 mask。
#   solver_generator_contract:
#     - 输入中恰有一个轴的镜像比较产生 1 个不匹配对，另一个轴不会也产生 1 个。
#     - 破缺对的两格共享同一相位，其中一格是局部 minority 旋转态，另一格是同相位 majority。
#     - minority 基础图案的主色 mask 关于目标轴对称、关于另一轴不对称，并且四个旋转态互异。
#     - 已给 active 集合既是完整 mask-active 集合的严格子集，又足以在轴居中约束下唯一定位。

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 600
AXES = ("vertical", "horizontal")
PHASE_SEQ = (0, 1, 2, 3, 2, 1, 0)


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


def _other_axis(axis: str) -> str:
    return "horizontal" if axis == "vertical" else "vertical"


def _axis_groups(axis: str) -> list[tuple[tuple[int, int], ...]]:
    if axis == "vertical":
        return [
            ((0, 0), (0, 2)),
            ((1, 0), (1, 2)),
            ((2, 0), (2, 2)),
            ((0, 1),),
            ((1, 1),),
            ((2, 1),),
        ]
    assert axis == "horizontal"
    return [
        ((0, 0), (2, 0)),
        ((0, 1), (2, 1)),
        ((0, 2), (2, 2)),
        ((1, 0),),
        ((1, 1),),
        ((1, 2),),
    ]


def _is_axis_symmetric(arr: np.ndarray, axis: str) -> bool:
    if axis == "vertical":
        return np.array_equal(arr, arr[:, ::-1])
    return np.array_equal(arr, arr[::-1, :])


def _phase_at(axis: str, i: int, j: int) -> int:
    idx = j if axis == "vertical" else i
    return int(PHASE_SEQ[idx])


def _apply_phase(arr: np.ndarray, phase: int) -> np.ndarray:
    return np.rot90(arr, int(phase) % 4)


def _flatten(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in arr.ravel())


def _rotation_signatures(arr: np.ndarray) -> set[tuple[int, ...]]:
    return {_flatten(_apply_phase(arr, phase)) for phase in range(4)}


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _centered_anchor(axis: str, n: int, mh: int, mw: int) -> tuple[int | None, int | None]:
    if axis == "vertical":
        assert (n - mw) % 2 == 0, "mask width must center on vertical axis"
        return None, (n - mw) // 2
    assert (n - mh) % 2 == 0, "mask height must center on horizontal axis"
    return (n - mh) // 2, None


def _axis_placement_candidates(mask: np.ndarray, existing: np.ndarray, axis: str) -> list[tuple[int, int]]:
    n = existing.shape[0]
    mh, mw = mask.shape
    fixed_top, fixed_left = _centered_anchor(axis, n, mh, mw)
    top_values = [fixed_top] if fixed_top is not None else range(0, n - mh + 1)
    left_values = [fixed_left] if fixed_left is not None else range(0, n - mw + 1)

    candidates: list[tuple[int, int]] = []
    for top in top_values:
        for left in left_values:
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    return candidates


def _mismatch_pairs(cell_pat: list[list[tuple[int, ...]]], axis: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    n = len(cell_pat)
    mismatches: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if axis == "vertical":
        for i in range(n):
            for j in range(n // 2):
                a = (i, j)
                b = (i, n - 1 - j)
                if cell_pat[a[0]][a[1]] != cell_pat[b[0]][b[1]]:
                    mismatches.append((a, b))
    else:
        for i in range(n // 2):
            for j in range(n):
                a = (i, j)
                b = (n - 1 - i, j)
                if cell_pat[a[0]][a[1]] != cell_pat[b[0]][b[1]]:
                    mismatches.append((a, b))
    return mismatches


def _family_presence(cell_pat: list[list[tuple[int, ...]]], base_arr: np.ndarray, axis: str) -> np.ndarray:
    n = len(cell_pat)
    phase_variants = {phase: _flatten(_apply_phase(base_arr, phase)) for phase in range(4)}
    present = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            present[i, j] = cell_pat[i][j] == phase_variants[_phase_at(axis, i, j)]
    return present


def _precompute_masks_for_axis(axis: str) -> dict[int, list[np.ndarray]]:
    groups = _axis_groups(axis)
    weights = [len(g) for g in groups]
    other_axis = _other_axis(axis)
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
        for k in range(1, len(groups) + 1):
            for combo in combinations(range(len(groups)), k):
                if sum(weights[i] for i in combo) != inactive_count:
                    continue
                mask = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
                for idx in combo:
                    for r, c in groups[idx]:
                        mask[r, c] = False
                if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
                    continue
                if not _is_axis_symmetric(mask.astype(int), axis):
                    continue
                if _is_axis_symmetric(mask.astype(int), other_axis):
                    continue
                if len(_rotation_signatures(mask.astype(int))) != 4:
                    continue
                masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_AXIS = {axis: _precompute_masks_for_axis(axis) for axis in AXES}


def _random_rotating_pattern(color_a: int, color_b: int) -> np.ndarray:
    while True:
        arr = np.array(
            [[random.choice((color_a, color_b)) for _ in range(MASK_SIZE)] for _ in range(MASK_SIZE)],
            dtype=int,
        )
        if np.all(arr == arr[0, 0]):
            continue
        if color_a not in arr or color_b not in arr:
            continue
        if len(_rotation_signatures(arr)) != 4:
            continue
        return arr


def _random_minority_mask(axis: str) -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_AXIS[axis][active_count])


def _build_grid(separator: int, majority_base: np.ndarray, minority_base: np.ndarray, axis: str, active_cells: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    active_set = set(active_cells)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            base = minority_base if (i, j) in active_set else majority_base
            paste(grid, _apply_phase(base, _phase_at(axis, i, j)), *_cell_origin(i, j))
    return grid


def _arrangement_mismatch_count(minority_cells: list[tuple[int, int]], axis: str) -> int:
    active = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for i, j in minority_cells:
        active[i, j] = True
    mismatches = 0
    if axis == "vertical":
        for i in range(CELL_GRID_N):
            for j in range(CELL_GRID_N // 2):
                if active[i, j] != active[i, CELL_GRID_N - 1 - j]:
                    mismatches += 1
    else:
        for i in range(CELL_GRID_N // 2):
            for j in range(CELL_GRID_N):
                if active[i, j] != active[CELL_GRID_N - 1 - i, j]:
                    mismatches += 1
    return mismatches


def _choose_given_subset(mask: np.ndarray, top: int, left: int, axis: str) -> tuple[tuple[int, int], ...] | None:
    active = [tuple(coord) for coord in np.argwhere(mask)]
    other_axis = _other_axis(axis)
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid: list[tuple[tuple[int, int], ...]] = []
        for subset in combinations(active, size):
            given_cells = [(top + r, left + c) for r, c in subset]
            if _arrangement_mismatch_count(given_cells, axis) != 1:
                continue
            if _arrangement_mismatch_count(given_cells, other_axis) == 1:
                continue
            existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            for i, j in given_cells:
                existing[i, j] = True
            if _axis_placement_candidates(mask, existing, axis) == [(top, left)]:
                valid.append(subset)
        if valid:
            return random.choice(valid)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the rotated minority family from the lone symmetry break and finish the fractal block.

    Input structure:
    - The board is a 7x7 grid of 3x3 cell interiors separated by 1-pixel divider lines.
    - Cell patterns are not uniform: along either rows or columns they follow a symmetric rotation gradient
      with phase sequence 0,1,2,3,2,1,0.
    - A 3x3 minority block is centered on that same mirror axis, but the input only shows a strict subset of
      its active cells, leaving exactly one mirrored pair mismatched.

    Rule:
    - Find the unique mirror axis whose paired cells disagree in exactly one place.
    - Take the two cell patterns in that broken pair, undo their shared local rotation, and test which one is
      the true minority base: its dominant-color mask must explain all same-family cells as a uniquely placed,
      axis-centered 3x3 block.
    - Rebuild the full block from that base mask, then rotate each completed cell back to its local phase.
    - Leave divider lines and all already-correct gradient tiles unchanged.

    Key insight: the exceptional cell is defined jointly by two global structures. It is the only tile that both
    belongs to the minority fractal family and breaks an otherwise coherent mirror-symmetric rotation gradient.
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
            r0, c0 = i * step + 1, j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = _flatten(pat_arr)
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)

    axis_hits: list[tuple[str, tuple[tuple[int, int], tuple[int, int]]]] = []
    for axis in AXES:
        mismatches = _mismatch_pairs(cell_pat, axis)
        if len(mismatches) == 1:
            axis_hits.append((axis, mismatches[0]))
    assert len(axis_hits) == 1, "expected exactly one mirror axis with a single broken pair"
    axis, (pa, pb) = axis_hits[0]
    phase = _phase_at(axis, *pa)
    assert phase == _phase_at(axis, *pb), "broken pair must share the same gradient phase"

    pat_a = cell_pat[pa[0]][pa[1]]
    pat_b = cell_pat[pb[0]][pb[1]]
    assert pat_a != pat_b, "broken mirror pair must contain two different cell patterns"

    solved: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for minority_pat, majority_pat in ((pat_a, pat_b), (pat_b, pat_a)):
        minority_base = _apply_phase(patterns[minority_pat], (-phase) % 4)
        majority_base = _apply_phase(patterns[majority_pat], (-phase) % 4)

        vals, cnts = np.unique(minority_base, return_counts=True)
        max_count = int(cnts.max())
        if max_count not in ACTIVE_PIXEL_COUNTS:
            continue
        active_color = int(vals[cnts.argmax()])
        mask = minority_base == active_color
        if not _is_axis_symmetric(mask.astype(int), axis):
            continue
        if _is_axis_symmetric(mask.astype(int), _other_axis(axis)):
            continue
        if len(_rotation_signatures(minority_base)) != 4:
            continue
        if _rotation_signatures(minority_base) & _rotation_signatures(majority_base):
            continue

        minority_cells = _family_presence(cell_pat, minority_base, axis)
        majority_cells = _family_presence(cell_pat, majority_base, axis)
        if not minority_cells[pa[0], pa[1]]:
            continue
        if not majority_cells[pb[0], pb[1]]:
            continue
        if np.any(minority_cells & majority_cells):
            continue
        if not np.all(minority_cells | majority_cells):
            continue
        if int(minority_cells.sum()) >= int(mask.sum()):
            continue

        candidates = _axis_placement_candidates(mask, minority_cells, axis)
        if len(candidates) == 1:
            top, left = candidates[0]
            solved.append((minority_base, majority_base, mask, top, left))

    assert len(solved) == 1, "broken pair must identify a unique phase-normalized minority family"
    minority_base, majority_base, mask, top, left = solved[0]

    out = g.copy()
    mh, mw = mask.shape
    for i in range(mh):
        for j in range(mw):
            gi, gj = top + i, left + j
            target_base = minority_base if mask[i, j] else majority_base
            rotated = _apply_phase(target_base, _phase_at(axis, gi, gj))
            r0, c0 = gi * step + 1, gj * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = rotated
    return out


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice(AXES)
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)

    majority_base = _random_rotating_pattern(shared, maj_exclusive)
    mask = _random_minority_mask(axis)
    minority_base = np.where(mask, min_exclusive, shared).astype(int)

    if len(_rotation_signatures(minority_base)) != 4:
        return None
    if _rotation_signatures(majority_base) & _rotation_signatures(minority_base):
        return None

    fixed_top, fixed_left = _centered_anchor(axis, CELL_GRID_N, MASK_SIZE, MASK_SIZE)
    top = fixed_top if fixed_top is not None else random.randrange(PLACEMENT_RANGE)
    left = fixed_left if fixed_left is not None else random.randrange(PLACEMENT_RANGE)

    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    if _arrangement_mismatch_count(active_cells, axis) != 0:
        return None
    if _arrangement_mismatch_count(active_cells, _other_axis(axis)) == 0:
        return None

    out = _build_grid(separator, majority_base, minority_base, axis, active_cells)

    subset = _choose_given_subset(mask, top, left, axis)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority_base, minority_base, axis, given_cells)

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
