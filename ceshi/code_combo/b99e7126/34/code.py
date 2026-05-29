# Combo variant of b99e7126 (index 34)
# combo_spec: D4/0+D10/0
# Name: 双向递归分形 + 递归分形缩放

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
#     - 保留“分隔线 + 等距单元格 + 单元格内部 3x3 图案”的原始骨架。
#     - 引入两个不同的少数单元格图案 B/C，它们都在自己的 3x3 内部编码一个基础掩码 M。
#     - 每一轮都不再做原题的一级 3x3 放大，而是做两级递归分形放大：kron(M, M)，得到 9x9 的单元格级窗口。
#     - 最终输出不是只补全一个窗口，而是先补全较稀少的 B 窗口，再在更新后的网格里检测并补全 C 窗口。
#     - 为了合法容纳两个互不重叠的 9x9 递归窗口，单元格网格从 7x7 / 13x13 扩大到 19x19。
#   behavior_trace:
#     - 原题：输入给出一个少数图案的若干单元格，输出补全同一 3x3 蓝图在大网格上的一次放大。
#     - D4 单变体：少数图案先复制成一级 3x3，再在每个活跃一级格里复制同一个 3x3，输出成为单个 9x9 自相似块。
#     - 本融合：输入同时含有两个不同的少数图案。第一轮先把更少见的图案补成一个 9x9 递归块；
#       得到中间网格后，仍然保留另一种未补全异常，于是第二轮再做一次同样的递归补全。
#     - 差异在于：求解者必须理解“每轮都是递归分形”以及“同一检测流程需要在更新后的网格上再次执行”。
#   complexity_calibration:
#     - 基础掩码只允许 3x3 且活跃数 5 或 6，避免 9x9 输出过密。
#     - 掩码要求四边都有活跃像素，且排除水平/垂直/旋转对称，减少平移歧义。
#     - 第一轮只暴露 2 个完整一级活跃子拷贝，第二轮暴露 3 个完整一级活跃子拷贝；这样既能看出递归，又能稳定控制顺序。
#     - 两个窗口互不重叠，避免不同异常在同一区域互相覆盖，保证视觉整洁。
#   implicit_assumption_audit:
#     - 原代码默认“只有 majority/minority 两类单元格图案”；这里会同时存在 majority、round1、round2 三类图案，因此必须按候选图案逐个检测。
#     - 原代码默认“最少见图案补完一次就结束”；这里需要循环执行，直到没有新的递归窗口可以继续补全。
#     - 原代码默认“最终待匹配窗口大小 == 基础掩码大小 3x3”；D4 后已经变成 9x9，本融合仍然沿用 9x9。
#     - 原代码默认“当前少数图案的全部已知位置都属于唯一一个窗口”；这里对 B 和 C 分别成立，generator 通过互不重叠的双窗口来保证。
#     - 原代码在窗口内把 inactive 位置写回 majority；这个前提在融合后仍需成立，因此 generator 必须保证另一类少数图案不会落入本轮窗口内部。
#   placement_legality:
#     - 19x19 单元格网格里放两个 9x9 窗口，要求矩形完全不重叠。
#     - 两个窗口都只覆盖完整单元格，不会越界，也不会产生半格覆盖。
#     - 输入只给每个窗口活跃位置的真子集，输出再补全剩余活跃格。
#   observability:
#     - 观察者先能在单个少数单元格内部看到 3x3 基础掩码，再能在更大尺度上看到同形的一级子拷贝，从而推断递归分形。
#     - 由于存在两种不同颜色的少数图案，且两块递归窗口分离，观察者还能看出求解流程不是一次完成，而是同一规则被执行两次。
#   visual_cleanliness:
#     - 画面只包含 divider、majority 图案、两种 minority 图案，没有额外标记层。
#     - 两个 9x9 递归块分区摆放，避免多层覆盖带来的混乱。
#   color_inference:
#     - 使用 5 色：divider、majority 专属色、shared 底色、round1 专属色、round2 专属色。
#     - 两种 minority 的活跃色直接可见，不依赖颜色编号大小等人类不可视信息。
#   solver_generator_contract:
#     - solver 新增假设：每个非 majority 图案若尚未完成，都能单独对应到一个唯一的 9x9 递归窗口。
#     - generator 通过“非重叠窗口 + 严格子集暴露 + 顺序计数控制 + main(inp)==out + 一轮/两轮输出差异检查”来保证这些假设。


CELL_INNER = 3
CELL_STEP = CELL_INNER + 1
MASK_SIZE = CELL_INNER
FRACTAL_LEVELS = 2
FRACTAL_SIZE = MASK_SIZE**FRACTAL_LEVELS
RECURSION_ROUNDS = 2


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


def _recursive_fractal_mask(mask: np.ndarray, levels: int = FRACTAL_LEVELS) -> np.ndarray:
    """Repeat a boolean mask inside each active cell for the requested recursion depth."""
    out = mask.astype(int)
    base = mask.astype(int)
    for _ in range(levels - 1):
        out = np.kron(out, base)
    return out.astype(bool)


def _extract_cell_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    inner = step - 1
    n = (g.shape[0] - 1) // step
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
    return patterns, cell_pat


def _count_patterns(cell_pat: list[list[tuple[int, ...]]]) -> dict[tuple[int, ...], int]:
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    return counts


def _base_mask_from_pattern(pat_arr: np.ndarray) -> np.ndarray:
    vals, cnts = np.unique(pat_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    return pat_arr == active_color


def _existing_positions(cell_pat: list[list[tuple[int, ...]]], target_pat: tuple[int, ...]) -> np.ndarray:
    n = len(cell_pat)
    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == target_pat
    return existing


def _unique_recursive_placement(existing: np.ndarray, fractal_mask: np.ndarray) -> tuple[int, int] | None:
    n = existing.shape[0]
    fh, fw = fractal_mask.shape
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
    if len(candidates) != 1:
        return None
    return candidates[0]


def _round_would_change(
    cell_pat: list[list[tuple[int, ...]]],
    top: int,
    left: int,
    fractal_mask: np.ndarray,
    minority_pat: tuple[int, ...],
    majority_pat: tuple[int, ...],
) -> bool:
    fh, fw = fractal_mask.shape
    for i in range(fh):
        for j in range(fw):
            desired = minority_pat if fractal_mask[i, j] else majority_pat
            if cell_pat[top + i][left + j] != desired:
                return True
    return False


def _apply_round(grid: np.ndarray) -> tuple[np.ndarray, bool]:
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    inner = step - 1
    patterns, cell_pat = _extract_cell_patterns(g, step)
    counts = _count_patterns(cell_pat)
    if len(counts) < 2:
        return g.copy(), False

    majority_pat = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    majority_arr = patterns[majority_pat]

    ordered_candidates = sorted(
        ((pat, cnt) for pat, cnt in counts.items() if pat != majority_pat),
        key=lambda kv: (kv[1], kv[0]),
    )
    for minority_pat, _ in ordered_candidates:
        minority_arr = patterns[minority_pat]
        base_mask = _base_mask_from_pattern(minority_arr)
        fractal_mask = _recursive_fractal_mask(base_mask)
        existing = _existing_positions(cell_pat, minority_pat)
        placement = _unique_recursive_placement(existing, fractal_mask)
        if placement is None:
            continue
        top, left = placement
        if not _round_would_change(cell_pat, top, left, fractal_mask, minority_pat, majority_pat):
            continue

        out = g.copy()
        fh, fw = fractal_mask.shape
        for i in range(fh):
            for j in range(fw):
                target_arr = minority_arr if fractal_mask[i, j] else majority_arr
                r0 = (top + i) * step + 1
                c0 = (left + j) * step + 1
                out[r0 : r0 + inner, c0 : c0 + inner] = target_arr
        return out, True

    return g.copy(), False


def _solve_rounds(grid: np.ndarray, max_rounds: int | None = None) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    rounds_done = 0
    round_cap = RECURSION_ROUNDS if max_rounds is None else max_rounds
    while rounds_done < round_cap:
        out, changed = _apply_round(out)
        if not changed:
            break
        rounds_done += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every uniquely placeable recursive fractal anomaly, one round after another.

    Input structure:
    - The canvas is a regular cell grid with 3x3 interiors separated by full divider rows and columns.
    - One majority cell pattern fills most cells, while one or more minority cell patterns appear as sparse anomalies.
    - Each minority pattern's own 3x3 interior acts as a base mask for a 9x9 recursive fractal window on the cell grid.

    Rule:
    - Repeatedly identify the least frequent incomplete non-majority pattern.
    - Convert its dominant-color 3x3 mask into a two-level recursive mask by embedding the mask inside every active cell of a first expansion.
    - Find the unique 9x9 placement consistent with the currently shown cells of that pattern, then overwrite that window with minority cells on active sites and majority cells on inactive sites.
    - After finishing one recursive window, inspect the updated grid again and repeat the same procedure for the next remaining anomaly.

    Key insight: The puzzle no longer ends after a single fractal completion. Each odd cell still contains a self-similar blueprint, but the completed grid may expose another sparse pattern that must undergo the same recursive expansion in a second pass.
    """
    return _solve_rounds(grid)


# --------------- generator (combo D4/0 + D10/0) ---------------

CELL_GRID_N = 19
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
PLACEMENT_RANGE = CELL_GRID_N - FRACTAL_SIZE + 1
ACTIVE_PIXEL_COUNTS = (5, 6)
ROUND1_MACRO_COPIES = 2
ROUND2_MACRO_COPIES = 3
COLOR_POOL = 10
MAX_RETRIES = 800


def _is_visibly_asymmetric(mask: np.ndarray) -> bool:
    """Reject masks whose recursive placement would be too symmetric and ambiguous."""
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
    """Enumerate 3x3 masks that stay legible after two-level recursive expansion."""
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
    arr = np.array(
        [random.choice((color_a, color_b)) for _ in range(MASK_SIZE * MASK_SIZE)],
        dtype=int,
    ).reshape(MASK_SIZE, MASK_SIZE)
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
    """Reveal full second-level copies for the selected active first-level macro-cells."""
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


def _choose_given_subset(
    base_mask: np.ndarray,
    fractal_mask: np.ndarray,
    top: int,
    left: int,
    macro_copy_count: int,
    max_extra_cells: int,
) -> list[tuple[int, int]] | None:
    """
    Pick a visible subset:
    - reveal several full second-level copies so the recursive motif is visible
    - optionally add a few anchor cells to force a unique placement
    """
    macro_active = _relative_active_positions(base_mask)
    fractal_active = _relative_active_positions(fractal_mask)
    if macro_copy_count >= len(macro_active):
        return None

    macro_combos = list(combinations(macro_active, macro_copy_count))
    random.shuffle(macro_combos)
    for macro_subset in macro_combos:
        base_subset = set(_fractal_subset_for_macro_positions(base_mask, macro_subset))
        remaining = [pos for pos in fractal_active if pos not in base_subset]

        extra_options: list[tuple[tuple[int, int], ...]] = [tuple()]
        if max_extra_cells >= 1:
            singles = [(pos,) for pos in remaining]
            random.shuffle(singles)
            extra_options.extend(singles)
        if max_extra_cells >= 2 and len(remaining) >= 2:
            pairs = list(combinations(remaining, 2))
            random.shuffle(pairs)
            extra_options.extend(pairs[:32])

        for extras in extra_options:
            subset = sorted(base_subset | set(extras))
            if _placement_candidates(fractal_mask, subset, top, left) == [(top, left)]:
                return subset
    return None


def _windows_overlap(a: tuple[int, int], b: tuple[int, int], size: int = FRACTAL_SIZE) -> bool:
    return not (
        a[0] + size <= b[0]
        or b[0] + size <= a[0]
        or a[1] + size <= b[1]
        or b[1] + size <= a[1]
    )


def _choose_two_disjoint_placements() -> tuple[tuple[int, int], tuple[int, int]] | None:
    placements = [(r, c) for r in range(PLACEMENT_RANGE) for c in range(PLACEMENT_RANGE)]
    random.shuffle(placements)
    for idx, first in enumerate(placements):
        for second in placements[idx + 1 :]:
            if not _windows_overlap(first, second):
                return first, second
    return None


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(
    separator: int,
    majority: np.ndarray,
    minority_specs: list[tuple[np.ndarray, list[tuple[int, int]]]],
) -> np.ndarray:
    """Construct the full grid by stamping majority everywhere, then minority subsets."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for minority, active_cells in minority_specs:
        for i, j in active_cells:
            paste(grid, minority, *_cell_origin(i, j))
    return grid


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, round1_color, round2_color = random.sample(range(COLOR_POOL), 5)
    majority = _random_two_color_pattern(shared, maj_exclusive)

    base_mask_1 = _random_minority_mask()
    base_mask_2 = _random_minority_mask()
    if np.array_equal(base_mask_1, base_mask_2):
        return None

    minority_1 = np.where(base_mask_1, round1_color, shared).astype(int)
    minority_2 = np.where(base_mask_2, round2_color, shared).astype(int)
    fractal_1 = _recursive_fractal_mask(base_mask_1)
    fractal_2 = _recursive_fractal_mask(base_mask_2)

    placements = _choose_two_disjoint_placements()
    if placements is None:
        return None
    (top1, left1), (top2, left2) = placements

    active_1 = [(top1 + r, left1 + c) for r, c in _relative_active_positions(fractal_1)]
    active_2 = [(top2 + r, left2 + c) for r, c in _relative_active_positions(fractal_2)]
    out = _build_grid(
        separator,
        majority,
        [
            (minority_1, active_1),
            (minority_2, active_2),
        ],
    )

    subset_1 = _choose_given_subset(base_mask_1, fractal_1, top1, left1, ROUND1_MACRO_COPIES, max_extra_cells=1)
    subset_2 = _choose_given_subset(base_mask_2, fractal_2, top2, left2, ROUND2_MACRO_COPIES, max_extra_cells=2)
    if subset_1 is None or subset_2 is None:
        return None
    if len(subset_1) >= len(subset_2):
        return None

    given_1 = [(top1 + r, left1 + c) for r, c in subset_1]
    given_2 = [(top2 + r, left2 + c) for r, c in subset_2]
    inp = _build_grid(
        separator,
        majority,
        [
            (minority_1, given_1),
            (minority_2, given_2),
        ],
    )

    try:
        one_round = _solve_rounds(inp, max_rounds=1)
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(one_round, solved):
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo-fractal b99e7126 input")
