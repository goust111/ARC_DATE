# Combo variant of b99e7126 (index 52)
# combo_spec: D1/3+D3/3+D7/1
# Name: 对称性破缺 + 多个相同异常 + 多次叠加

from __future__ import annotations

from functools import reduce
from itertools import combinations
from math import gcd
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import paste


"""
design_analysis:
  a_concretization_and_trace:
    - 原题在 7x7 单元格网格里只找 1 个 minority 单元，再把它的 3x3 内部图案放大成 1 个
      3x3 单元块并补全。
    - 本融合版先把整张图约束成关于某条镜像轴近乎对称；majority 与 minority 单元内部本身也都
      关于该轴对称。输入里只在轴的一侧放出 2-3 个相同 minority 单元，它们都缺少镜像伙伴，
      因而形成多个同类 symmetry breaks。
    - 每个 break 不是只提示“这里有个特殊格”，而是各自对应同一个 3x3 放大块的一个固定局部坐标。
      solver 先识别镜像轴，再从 minority 图案的 active mask 中枚举哪个局部坐标被拿来当作输入锚点；
      一旦局部坐标确定，每个异常格都给出一个完整放大块的位置。
    - 输出把 minority 图案在这些块位置全部叠加；若两个块重叠，则按 active 并集保留 minority。
    - 具体小例子（竖直轴）：
        1. minority mask 关于竖直轴对称，输入只显示 local=(2,0) 的两个实例，落在全局
           (2,2) 与 (6,2)；它们的镜像位置 (2,4)、(6,4) 仍是 majority，于是竖直轴上出现 2 个破缺点。
        2. solver 识别轴后发现这两个异常都只能解释为“某个轴居中 3x3 块的左下 active 像素”。
        3. 因而恢复出 top=0 和 top=4 两个放大块，并把两块的 active 单元并起来写成输出。
    - 与原题差异：
        1. 不再靠“最少数图案”定位唯一种子，而是靠多个同类破缺共同指向镜像轴与锚点角色。
        2. 不再只补一个放大块，而是每个破缺点都触发一次同构放大块叠加。
        3. 输入里的 minority 单元不再是某个单块 active 集合的真子集，而是多个块各取 1 个同位锚点。
  b_complexity_calibration:
    - 轴只在竖直 / 水平二选一，避免镜像与旋转混杂。
    - 放大块仍固定为 3x3，minority active 数仍限制在 6-7，和原题的视觉密度接近。
    - 每个样本只用 2-3 个异常锚点，并要求至少一对块发生重叠，这样“多次叠加”既明显又不会铺满全图。
    - 所有异常都使用同一个局部坐标；这让人类能从多组样例里观察到“同一种偏轴缺口重复出现”，
      同时保持 solver 的搜索空间小而稳定。
  c_implicit_assumption_audit:
    - 原 main() 的 `minority_pat = min(counts...)` 依赖“唯一少数图案就是目标”。融合后虽然 minority
      仍可能更少，但真正的识别依据改成“所有 minority 都是镜像破缺点”；solver 直接从 mismatch 对中
      提取候选图案，而不依赖频数。
    - 原 main() 假设所有已给 minority 都属于同一个 3x3 放大块，因此会搜索 1 个 `(top,left)`。
      融合后这个前提失效；新 solver 要从所有已给 minority 反推出多个 anchors。
    - 原 main() 顺序写单个块时不存在覆盖冲突。融合后多个块可重叠，所以必须先求所有块 active 的并集，
      再一次性着色，不能让某块的 inactive 覆盖另一块的 active。
    - 新增假设是：输入中的每个 minority 都是同一个局部 seed 坐标的实例，而且它们的镜像对应位置全是
      majority。generator 用构造保证这一点。
  d_placement_legality:
    - 镜像轴居中意味着竖直轴时所有块 `left=2`，水平轴时所有块 `top=2`；3x3 块始终在 7x7 内。
    - anchors 只取 0..4 的合法 top 或 left；输入异常由这些合法块里的某个 active 像素投影而来。
    - 多块可重叠，但最终只保留 active 并集，不会出现第三种覆盖产物。
    - generator 采用 output-first：先决定轴、mask、seed、anchors 和输出，再生成输入，并用新 main()
      回算校验。
  e_visual_observability:
    - 人眼能看到一条明显的镜像轴，因为 majority / minority 的单元内部都对该轴自对称。
    - 输入里的少数异常格都位于轴同一侧、形状完全相同，且各自都缺少镜像伙伴；这就是 D1 和 D7 的可见线索。
    - 输出则把每个破缺点扩展成一个完整 3x3 放大块，多个块重叠后形成更大的对称结构，清楚体现 D3。
  f_visual_cleanliness:
    - 全图只有 separator / majority / minority 三层对象角色，没有 decoy 或额外标记。
    - 叠加只发生在同一条轴向带内，且仍然只在 majority/minority 两种单元纹理之间切换，画面不会失控。
  g_inferability:
    - 颜色只承担“区分 divider / majority / minority”与 minority 内部 active mask 的作用，不依赖颜色值排序。
    - 规则由结构支撑：镜像轴、同位缺口、局部 seed 坐标、多块并集。跨样例随机的是轴方向、seed 行列、
      anchors 位置与具体颜色，但可归纳的空间关系保持稳定。
"""


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
INPUT_ANCHOR_COUNTS = (2, 3)
COLOR_POOL = 10
MAX_RETRIES = 600
AXES = ("vertical", "horizontal")


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
    return np.array_equal(arr[::-1, :], arr)


def _mirror_pos(pos: tuple[int, int], axis: str, n: int) -> tuple[int, int]:
    r, c = pos
    if axis == "vertical":
        return r, n - 1 - c
    return n - 1 - r, c


def _centered_origin(axis: str, n: int, mh: int, mw: int) -> tuple[int | None, int | None]:
    if axis == "vertical":
        assert (n - mw) % 2 == 0
        return None, (n - mw) // 2
    assert (n - mh) % 2 == 0
    return (n - mh) // 2, None


def _extract_patterns(
    g: np.ndarray, step: int, inner: int
) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    n = (g.shape[0] - 1) // step
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(int(v) for v in pat_arr.ravel())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)
    return patterns, cell_pat


def _mismatch_pairs(
    cell_pat: list[list[tuple[int, ...]]], axis: str
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
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


def _active_mask_from_pattern(pattern: np.ndarray) -> np.ndarray:
    vals, cnts = np.unique(pattern, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    return pattern == active_color


def _off_axis_active_coords(mask: np.ndarray, axis: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    mid = mask.shape[1] // 2 if axis == "vertical" else mask.shape[0] // 2
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if axis == "vertical":
            if cc != mid:
                out.append((rr, cc))
        else:
            if rr != mid:
                out.append((rr, cc))
    return out


def _anchors_from_existing(
    existing_cells: list[tuple[int, int]],
    seed: tuple[int, int],
    axis: str,
    n: int,
    mh: int,
    mw: int,
) -> list[tuple[int, int]] | None:
    seed_r, seed_c = seed
    fixed_top, fixed_left = _centered_origin(axis, n, mh, mw)
    anchors: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for gr, gc in existing_cells:
        if axis == "vertical":
            assert fixed_left is not None
            if gc != fixed_left + seed_c:
                return None
            top = gr - seed_r
            left = fixed_left
        else:
            assert fixed_top is not None
            if gr != fixed_top + seed_r:
                return None
            top = fixed_top
            left = gc - seed_c
        if not (0 <= top <= n - mh and 0 <= left <= n - mw):
            return None
        anchor = (int(top), int(left))
        if anchor in seen:
            return None
        seen.add(anchor)
        anchors.append(anchor)
    anchors.sort()
    return anchors


def _union_from_anchors(mask: np.ndarray, anchors: list[tuple[int, int]], n: int) -> np.ndarray:
    mh, mw = mask.shape
    union = np.zeros((n, n), dtype=bool)
    for top, left in anchors:
        union[top : top + mh, left : left + mw] |= mask
    return union


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    active_cells: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _solve_candidates(
    g: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], tuple[int, int], str]]:
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1
    patterns, cell_pat = _extract_patterns(g, step, inner)

    solutions: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], tuple[int, int], str]] = []
    for axis in AXES:
        mismatches = _mismatch_pairs(cell_pat, axis)
        if len(mismatches) < 2:
            continue
        mismatch_patterns = {
            cell_pat[a[0]][a[1]]
            for pair in mismatches
            for a in pair
        }
        if len(mismatch_patterns) != 2:
            continue
        pat_a, pat_b = tuple(mismatch_patterns)
        for minority_pat, majority_pat in ((pat_a, pat_b), (pat_b, pat_a)):
            minority_arr = patterns[minority_pat]
            majority_arr = patterns[majority_pat]
            mask = _active_mask_from_pattern(minority_arr)
            if mask.shape != (inner, inner):
                continue
            if int(mask.sum()) not in ACTIVE_PIXEL_COUNTS:
                continue
            if not _is_axis_symmetric(mask.astype(int), axis):
                continue

            existing_cells: list[tuple[int, int]] = []
            valid_existing = True
            for i in range(n):
                for j in range(n):
                    if cell_pat[i][j] == minority_pat:
                        mirror = _mirror_pos((i, j), axis, n)
                        if cell_pat[mirror[0]][mirror[1]] != majority_pat:
                            valid_existing = False
                            break
                        existing_cells.append((i, j))
                if not valid_existing:
                    break
            if not valid_existing or len(existing_cells) < 2:
                continue

            for seed in _off_axis_active_coords(mask, axis):
                anchors = _anchors_from_existing(existing_cells, seed, axis, n, mask.shape[0], mask.shape[1])
                if anchors is None:
                    continue
                solutions.append((minority_arr, majority_arr, mask, anchors, seed, axis))
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a symmetric multi-overlay by treating every symmetry-breaking odd cell as the same repeated anchor role.

    Input structure:
    - The grid is a 7x7 array of 3x3 interior cells split by 1-pixel divider lines.
    - Most cells share one axis-symmetric majority interior.
    - A small set of identical minority cells appears only on one side of a global mirror axis, so each one lacks
      its mirror partner and acts as a symmetry break.

    Rule:
    - Find the mirror axis by examining mirrored cell pairs and isolate the two patterns involved in the broken pairs.
    - Use the minority cell's dominant-color mask as a 3x3 stencil, then determine which off-axis active pixel role
      every shown minority cell is playing.
    - Each shown minority cell defines one axis-centered 3x3 overlay block; place that block at every recovered anchor.
    - Combine all recovered blocks with active-union semantics: a cell becomes minority if any overlay marks it active,
      otherwise it remains majority.

    Key insight: the odd cells are not a partial copy of one enlarged motif. They are multiple copies of the same
    symmetry-breaking anchor, and each one independently requests the same enlarged motif at its own aligned position.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    solved = _solve_candidates(g)
    assert len(solved) == 1, "expected exactly one symmetry-guided multi-overlay interpretation"
    minority_arr, majority_arr, mask, anchors, _seed, _axis = solved[0]

    union = _union_from_anchors(mask, anchors, n)
    out = g.copy()
    for i in range(n):
        for j in range(n):
            target = minority_arr if union[i, j] else majority_arr
            r0 = i * step + 1
            c0 = j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target
    return out


def _precompute_masks_for_axis(axis: str) -> dict[int, list[np.ndarray]]:
    groups = _axis_groups(axis)
    weights = [len(group) for group in groups]
    other_axis = _other_axis(axis)
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
        for k in range(1, len(groups) + 1):
            for combo in combinations(range(len(groups)), k):
                if sum(weights[idx] for idx in combo) != inactive_count:
                    continue
                mask = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
                for idx in combo:
                    for r, c in groups[idx]:
                        mask[r, c] = False
                if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
                    continue
                if _is_axis_symmetric(mask.astype(int), other_axis):
                    continue
                if not _off_axis_active_coords(mask, axis):
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
            flip = color_b if choices[0] == color_a else color_a
            choices[random.randrange(len(choices))] = flip
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


def _seed_options(mask: np.ndarray, axis: str) -> list[tuple[int, int]]:
    mid = mask.shape[1] // 2 if axis == "vertical" else mask.shape[0] // 2
    seeds: list[tuple[int, int]] = []
    for r, c in _off_axis_active_coords(mask, axis):
        if axis == "vertical":
            if c < mid:
                seeds.append((r, c))
        else:
            if r < mid:
                seeds.append((r, c))
    if seeds:
        return seeds
    return _off_axis_active_coords(mask, axis)


def _input_cells_from_anchors(
    anchors: list[tuple[int, int]], seed: tuple[int, int]
) -> list[tuple[int, int]]:
    sr, sc = seed
    return sorted((top + sr, left + sc) for top, left in anchors)


def _overlap_score(anchors: list[tuple[int, int]]) -> int:
    score = 0
    for i, (ta, la) in enumerate(anchors):
        for tb, lb in anchors[i + 1 :]:
            if max(abs(ta - tb), abs(la - lb)) < MASK_SIZE:
                score += 1
    return score


def _choose_anchors(axis: str) -> list[tuple[int, int]]:
    fixed_top, fixed_left = _centered_origin(axis, CELL_GRID_N, MASK_SIZE, MASK_SIZE)
    assert (fixed_top is None) != (fixed_left is None)
    free_values = list(range(PLACEMENT_RANGE))
    combos: list[list[tuple[int, int]]] = []
    for size in INPUT_ANCHOR_COUNTS:
        for combo in combinations(free_values, size):
            if max(combo) - min(combo) < MASK_SIZE:
                continue
            anchors: list[tuple[int, int]] = []
            for value in combo:
                if axis == "vertical":
                    anchors.append((int(value), int(fixed_left)))
                else:
                    anchors.append((int(fixed_top), int(value)))
            if _overlap_score(anchors) == 0:
                continue
            combos.append(anchors)
    assert combos, "expected at least one legal anchor combination"
    return random.choice(combos)


def _single_block_legacy_candidates(mask: np.ndarray, shown_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in shown_cells:
        existing[r, c] = True
    candidates: list[tuple[int, int]] = []
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


def _type_mismatch_count(active_cells: list[tuple[int, int]], axis: str) -> int:
    active = set(active_cells)
    mismatches = 0
    if axis == "vertical":
        for i in range(CELL_GRID_N):
            for j in range(CELL_GRID_N // 2):
                if ((i, j) in active) != ((i, CELL_GRID_N - 1 - j) in active):
                    mismatches += 1
    else:
        for i in range(CELL_GRID_N // 2):
            for j in range(CELL_GRID_N):
                if ((i, j) in active) != ((CELL_GRID_N - 1 - i, j) in active):
                    mismatches += 1
    return mismatches


def _gen_attempt() -> tuple[np.ndarray, dict[str, int | str]] | None:
    axis = random.choice(AXES)
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)

    majority = _random_axis_pattern(axis, shared, maj_exclusive)
    mask = _random_minority_mask(axis)
    minority = np.where(mask, min_exclusive, shared).astype(int)
    if np.array_equal(majority, minority):
        return None

    seed = random.choice(_seed_options(mask, axis))
    anchors = _choose_anchors(axis)
    shown_cells = _input_cells_from_anchors(anchors, seed)

    union = _union_from_anchors(mask, anchors, CELL_GRID_N)
    active_cells = sorted((int(r), int(c)) for r, c in np.argwhere(union))
    if len(active_cells) <= len(shown_cells):
        return None
    if _type_mismatch_count(shown_cells, axis) != len(shown_cells):
        return None
    if _single_block_legacy_candidates(mask, shown_cells):
        return None

    inp = _build_grid(separator, majority, minority, shown_cells)
    out = _build_grid(separator, majority, minority, active_cells)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    info = {
        "axis": axis,
        "anchors": len(anchors),
        "shown": len(shown_cells),
        "union_active": len(active_cells),
        "overlap_pairs": _overlap_score(anchors),
        "legacy_single_candidates": len(_single_block_legacy_candidates(mask, shown_cells)),
        "mismatch_count": _type_mismatch_count(shown_cells, axis),
    }
    return inp, info


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
