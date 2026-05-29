# Variant of b99e7126
# Dimension: D1 — 匹配条件：异常单元格的识别方式 (match_cond)
# Change: 通过比较所有单元格的内部颜色图案，找到与多数单元格图案不同的那个唯一异常单元格... → 对称性破缺检测：找打破对称性的单元格

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
#     - 保留原题的 29x29 画布、7x7 单元格网格、3x3 单元格内部、以及“把某个 3x3
#       单元格图案放大成 3x3 个单元格”的分形补全过程。
#   core_D1_change:
#     - 原题靠“哪个单元格图案出现次数最少”来找种子单元格。
#     - 本变体改成“先识别整张网格的全局镜像轴，再找唯一缺少镜像伙伴的特殊单元格”。
#   concretization:
#     - 选择一个全局镜像轴：竖直或水平。
#     - 输出中的 minority 单元格块必须关于该轴对称，因此 3x3 分形块被固定在轴中心：
#       竖直轴时 left=2，水平轴时 top=2；另一个方向仍保留 0..4 的位移自由度。
#     - 输入只显示 3-4 个 minority 单元格；其中绝大部分满足镜像配对，但恰好有 1 个
#       偏轴 minority 单元格没有出现它的镜像伙伴，因此整张网格只破坏一次镜像对称。
#     - 为了防止任务退化回“直接找最少数图案”，额外加入一对完全对称的 decoy 单元格。
#       这样输入中至少有三类单元格：majority / minority / decoy，且 decoy 往往更少见。
#   behavior_trace:
#     - 原逻辑：
#         1. 统计每种单元格图案频数。
#         2. 取最少数图案作为 minority。
#         3. 用它的内部主色生成 3x3 mask，再对齐已有 minority 位置完成输出。
#     - 新逻辑：
#         1. 比较镜像成对的单元格，找到“只有一个不匹配对”的轴。
#         2. 该不匹配对里，一边是 minority，一边是 majority。
#         3. 依次尝试把这两个图案当作分形种子；只有真正的 minority 图案，其所有出现位置
#            才能作为一个轴居中的 3x3 mask 的严格子集，并得到唯一放置。
#         4. 按该 mask 补齐输出。
#     - 具体小例子（竖直轴）：
#         输入在全局第 3 列为镜像轴；某一行的 (r,2) 是 minority，但 (r,4) 仍是 majority。
#         其他 minority 要么成对出现，要么落在轴上单独自对称。于是只有这一对破坏了镜像。
#   complexity_calibration:
#     - 只使用两种镜像轴（竖直 / 水平），避免引入旋转、多种规则混杂。
#     - 仍保留原题的 3x3 mask、6-7 个激活像素、3-4 个已给 minority 线索。
#     - decoy 只放 1 对，既能打破“少数计数”假设，又不会让画面过于杂乱。
#   implicit_assumption_audit:
#     - 原 main() 里 `minority_pat = min(counts...)` 依赖“唯一少数图案 = 目标图案”；
#       本变体不成立，因为 decoy 可能更少。
#     - 原 main() 的自由放置搜索默认 3x3 块可在任意 top/left 放置；本变体中镜像轴会固定
#       一个坐标（竖直轴固定 left=2，水平轴固定 top=2）。
#     - 原题默认所有非-majority 特殊单元格都参与分形；本变体不成立，因为 decoy 必须忽略。
#   legality:
#     - 轴居中的 3x3 块在 7x7 内始终合法。
#     - decoy 只放在分形 3x3 块外，并以镜像对形式出现，不越界、不重叠、不被后续覆盖。
#   visual_observability:
#     - 输出是整张网格恢复成完整镜像结构；输入只缺一个镜像伙伴，人眼可直接看到“哪边少了一个”。
#     - decoy 成对放置，给了解题者额外的镜像支撑，帮助识别真正的全局轴，而不是靠颜色 id。
#   cleanliness:
#     - 所有特殊元素都还是 3x3 单元格图案，没有额外叠层。
#     - decoy 数量固定为 2 个，且不进入分形块内部，避免覆盖混乱。
#   color_inference:
#     - 颜色只承担“不同图案”的作用，不参与排序或运算。
#     - minority / majority / decoy 各自都有稳定的形状支持，解题者不需要记忆颜色编号。
#   solver_generator_contract:
#     - 输入上恰有一个镜像轴产生 1 对不匹配单元格，另一个轴不会也产生 1 对不匹配。
#     - 那对不匹配里包含真正的 minority 与 majority。
#     - minority 图案的主色 mask 为 3x3、激活 6-7 格、覆盖四条边。
#     - 已给 minority 位置是该 mask 的严格子集，并且恰好包含 1 个未配对的偏轴格。


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


def _mirror_pos(pos: tuple[int, int], axis: str, n: int) -> tuple[int, int]:
    r, c = pos
    if axis == "vertical":
        return r, n - 1 - c
    return n - 1 - r, c


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the lone symmetry-breaking special cell as the seed for the same fractal completion.

    Input structure:
    - Grid divided into a 7x7 array of 3x3 cells by 1-pixel separator lines.
    - The whole picture is almost mirror-symmetric about either the vertical or horizontal midline.
      Most cells are the majority pattern, one decoy pattern appears in a clean mirrored pair, and a few
      minority cells appear as a partial 3x3 fractal block with exactly one off-axis cell missing its mirror mate.

    Rule:
    - Detect the intended mirror axis by finding the only axis with exactly one mismatched mirrored cell pair.
    - From that mismatched pair, determine which side is the true fractal seed: its interior dominant-color mask
      must fit all cells of that type as a uniquely placed, axis-centered 3x3 block.
    - Use that seed cell's 3x3 mask at cell scale: mask-True positions become the seed pattern, and mask-False
      positions inside the placed block become the paired majority pattern.
    - Leave the separator lines and the symmetric decoy cells unchanged.

    Key insight: the target is no longer “the rarest tile”. It is the only special tile that breaks an otherwise
    coherent mirror layout, and that symmetry break identifies both the correct seed and the centered placement.
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

    axis_hits: list[tuple[str, tuple[tuple[int, int], tuple[int, int]]]] = []
    for axis in ("vertical", "horizontal"):
        mismatches = _mismatch_pairs(cell_pat, axis)
        if len(mismatches) == 1:
            axis_hits.append((axis, mismatches[0]))
    assert len(axis_hits) == 1, "expected exactly one mirror axis with a single broken pair"
    axis, (pa, pb) = axis_hits[0]

    pat_a = cell_pat[pa[0]][pa[1]]
    pat_b = cell_pat[pb[0]][pb[1]]
    assert pat_a != pat_b, "broken mirror pair must contain two different cell patterns"

    solved: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for minority_pat, majority_pat in ((pat_a, pat_b), (pat_b, pat_a)):
        minority_arr = patterns[minority_pat]
        vals, cnts = np.unique(minority_arr, return_counts=True)
        active_color = int(vals[cnts.argmax()])
        mask = minority_arr == active_color

        existing = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                existing[i, j] = cell_pat[i][j] == minority_pat

        if int(existing.sum()) >= int(mask.sum()):
            continue

        candidates = _axis_placement_candidates(mask, existing, axis)
        if len(candidates) == 1:
            top, left = candidates[0]
            solved.append((minority_arr, patterns[majority_pat], mask, top, left))

    assert len(solved) == 1, "mirror-broken pair must identify a unique fractal seed"
    minority_arr, majority_arr, mask, top, left = solved[0]

    out = g.copy()
    mh, mw = mask.shape
    for i in range(mh):
        for j in range(mw):
            target_pat = minority_arr if mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (symmetry-break variant) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 400
AXES = ("vertical", "horizontal")


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
                if _is_axis_symmetric(mask.astype(int), other_axis):
                    continue
                masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_AXIS = {axis: _precompute_masks_for_axis(axis) for axis in AXES}


def _random_axis_pattern(axis: str, color_a: int, color_b: int) -> np.ndarray:
    groups = _axis_groups(axis)
    other_axis = _other_axis(axis)
    while True:
        arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
        choices = [random.choice((color_a, color_b)) for _ in groups]
        if all(v == choices[0] for v in choices):
            choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
        for group, color in zip(groups, choices):
            for r, c in group:
                arr[r, c] = color
        if not _is_axis_symmetric(arr, axis):
            continue
        if _is_axis_symmetric(arr, other_axis):
            continue
        return arr


def _random_minority_mask(axis: str) -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_AXIS[axis][active_count])


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    decoy: np.ndarray,
    active_cells: list[tuple[int, int]],
    decoy_cells: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in decoy_cells:
        paste(grid, decoy, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _type_grid(minority_cells: list[tuple[int, int]], decoy_cells: list[tuple[int, int]]) -> np.ndarray:
    types = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=int)
    for i, j in decoy_cells:
        types[i, j] = 2
    for i, j in minority_cells:
        assert types[i, j] == 0, "minority / decoy overlap"
        types[i, j] = 1
    return types


def _arrangement_mismatch_count(minority_cells: list[tuple[int, int]], decoy_cells: list[tuple[int, int]], axis: str) -> int:
    types = _type_grid(minority_cells, decoy_cells)
    mismatches = 0
    if axis == "vertical":
        for i in range(CELL_GRID_N):
            for j in range(CELL_GRID_N // 2):
                if types[i, j] != types[i, CELL_GRID_N - 1 - j]:
                    mismatches += 1
    else:
        for i in range(CELL_GRID_N // 2):
            for j in range(CELL_GRID_N):
                if types[i, j] != types[CELL_GRID_N - 1 - i, j]:
                    mismatches += 1
    return mismatches


def _subset_break_count(subset: tuple[tuple[int, int], ...], axis: str) -> int:
    shown = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    for r, c in subset:
        shown[r, c] = True
    if axis == "vertical":
        return int(np.sum(shown[:, 0] != shown[:, 2]))
    return int(np.sum(shown[0, :] != shown[2, :]))


def _choose_decoy_pair(axis: str, block_cells: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    candidates: list[list[tuple[int, int]]] = []
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            a = (i, j)
            b = _mirror_pos(a, axis, CELL_GRID_N)
            if a == b or a > b:
                continue
            if axis == "vertical" and i == CELL_GRID_N // 2:
                continue
            if axis == "horizontal" and j == CELL_GRID_N // 2:
                continue
            pair = [a, b]
            if any(cell in block_cells for cell in pair):
                continue
            candidates.append(pair)
    if not candidates:
        return None
    return random.choice(candidates)


def _choose_given_subset(
    mask: np.ndarray,
    top: int,
    left: int,
    axis: str,
    decoy_cells: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    other_axis = _other_axis(axis)
    for size in sizes:
        valid: list[tuple[tuple[int, int], ...]] = []
        for subset in combinations(active, size):
            if _subset_break_count(subset, axis) != 1:
                continue
            given_cells = [(top + r, left + c) for r, c in subset]
            if _arrangement_mismatch_count(given_cells, decoy_cells, axis) != 1:
                continue
            if _arrangement_mismatch_count(given_cells, decoy_cells, other_axis) == 1:
                continue

            existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            for r, c in given_cells:
                existing[r, c] = True
            if _axis_placement_candidates(mask, existing, axis) == [(top, left)]:
                valid.append(subset)
        if valid:
            return random.choice(valid)
    return None


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice(AXES)
    separator, maj_exclusive, shared, min_exclusive, decoy_exclusive = random.sample(range(COLOR_POOL), 5)

    majority = _random_axis_pattern(axis, shared, maj_exclusive)
    mask = _random_minority_mask(axis)
    minority = np.where(mask, min_exclusive, shared).astype(int)
    decoy = _random_axis_pattern(axis, shared, decoy_exclusive)

    if np.array_equal(majority, minority) or np.array_equal(majority, decoy) or np.array_equal(minority, decoy):
        return None

    fixed_top, fixed_left = _centered_anchor(axis, CELL_GRID_N, MASK_SIZE, MASK_SIZE)
    top = fixed_top if fixed_top is not None else random.randrange(CELL_GRID_N - MASK_SIZE + 1)
    left = fixed_left if fixed_left is not None else random.randrange(CELL_GRID_N - MASK_SIZE + 1)

    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    block_cells = {(top + r, left + c) for r in range(MASK_SIZE) for c in range(MASK_SIZE)}
    decoy_cells = _choose_decoy_pair(axis, block_cells)
    if decoy_cells is None:
        return None

    if _arrangement_mismatch_count(active_cells, decoy_cells, axis) != 0:
        return None
    if _arrangement_mismatch_count(active_cells, decoy_cells, _other_axis(axis)) == 0:
        return None

    out = _build_grid(separator, majority, minority, decoy, active_cells, decoy_cells)

    subset = _choose_given_subset(mask, top, left, axis, decoy_cells)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, minority, decoy, given_cells, decoy_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 D1 input")
