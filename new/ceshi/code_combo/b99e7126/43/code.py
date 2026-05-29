# Combo variant of b99e7126 (index 43)
# combo_spec: D4/0+D11/2
# Name: 多色差异化分形 + 递归分形缩放

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
#     - 仍保留“分隔线 + 规则单元格网格 + 多数背景单元格”的原题骨架，但把异常图案升级成三态模板：
#         0 = shared 背景像素
#         1 = 递归色像素
#         2 = 浅层色像素
#     - 3x3 异常单元格本身就是模板 T。输出不再是简单 `kron(mask, mask)`，而是做一轮带类型的替换：
#         * T 中的递归色格，在大尺度上展开成一整份 T
#         * T 中的浅层色格，在大尺度上只展开成一个实心 3x3 浅层块
#         * T 中的 shared 格保持多数单元格
#       因而最终宏图案仍是 9x9，但包含“递归分支”和“停止递归的实心块”两种行为。
#     - 为了容纳 9x9 宏图案并允许平移，单元格网格沿用递归单变体的 13x13，整张像素网格是 53x53。
#     - 单元格图案固定为 3 类：
#         * majority: 两色随机纹理 `{shared, maj_exclusive}`
#         * recursive/blueprint: 三色模板 `{shared, recur_color, shallow_color}`
#         * shallow: 单色实心 `{shallow_color}`
#   behavior_trace:
#     - 原题：异常单元格里“活跃像素”放大后全部变成同一种 minority 单元格。
#     - D4 单变体：活跃像素先放成一级宏块，再在每个活跃宏格里嵌入同一份模板。
#     - 本融合：模板里的两种非 shared 颜色分家。
#         例：若模板 T 的某个位置是 recur_color，则对应 3x3 宏格会出现一整份 T；
#             若模板 T 的某个位置是 shallow_color，则对应 3x3 宏格只会被浅层色实心填满。
#       所以观察者会同时看到“同一份三色模板在大尺度重复”以及“其中某种颜色只扩一级”的层级差异。
#   complexity_calibration:
#     - 基础模板只允许 4-5 个非 shared 像素，其中 recur 2-3 个、shallow 1-2 个，避免 9x9 窗口过满。
#     - 模板要求占据四条边且拒绝旋转/镜像对称，降低放置歧义。
#     - 输入至少显式展示：
#         * 1 份完整递归子拷贝（让人看出模板在大尺度重现）
#         * 1 个完整浅层 3x3 实心块（让人看出 shallow_color 只扩一级）
#       需要时再补少量锚点来打破平移歧义。
#   implicit_assumption_audit:
#     - 原 `len(counts) >= 2` 只区分 majority/minority；现在 solver 改成恰有 3 种单元格图案。
#     - 原逻辑把 minority 的“最高频颜色”视为唯一活跃色；现在必须保留 blueprint 内两种非 shared 颜色。
#     - 原候选匹配只检查布尔 mask；现在要匹配 9x9 typed layout，区分 recursive 单元格与 shallow 单元格。
#     - 原填充循环只在 majority/minority 二选一；现在变成 0/1/2 三选一。
#     - 新隐式假设是：唯一三色单元格就是 blueprint，唯一单色单元格就是 shallow；generator 用固定图案族保证它们存在且可区分。
#   placement_legality:
#     - 9x9 宏图案放入 13x13 单元格网格，平移范围仍是 0..4，不越界。
#     - 顶层递归块和浅层实心块都严格占用整格；不存在半格覆盖或不同对象重叠。
#     - subset 构造只从目标 9x9 宏图案中抽取真子集，所以输入永远是合法的不完整输出。
#   observability:
#     - 人眼能先在单个 recursive 单元格里看到三色模板 T，再在输入里看到：
#         * 某处 3x3 子窗复现同一份 T
#         * 某处 3x3 子窗被 shallow_color 整块铺满
#       于是可以归纳出“不同颜色在递归时分工不同”。
#     - shallow 单元格是纯色，使得“blueprint 中的某个颜色 -> 大尺度实心块”的映射直接可见。
#   visual_cleanliness:
#     - 仍只有 divider / majority / recursive / shallow 四层视觉元素，没有额外标记。
#     - 递归拷贝与浅层实心块都被约束在同一个 9x9 方窗内，结构整齐，层次清楚。
#   color_inference:
#     - 只用 5 色：separator、maj_exclusive、shared、recur_color、shallow_color。
#     - 颜色角色通过结构重复显式给出：
#         * recursive 单元格复现整份模板
#         * shallow 单元格本身就是 pure `shallow_color`
#         * shared 颜色可由 majority 与 blueprint 的交集直接识别
#   solver_generator_contract:
#     - solver 假设输入满足：
#         * 恰有 3 种单元格图案
#         * 三色图案是 blueprint，单色图案是 shallow
#         * 现有非多数单元格是 9x9 typed fractal 的真子集，且放置唯一
#     - generator 通过固定图案家族、唯一候选校验、`main(inp) == out` 验证，以及后续融合度统计来保证这些前提成立。


CELL_INNER = 3
CELL_STEP = 4
FRACTAL_LEVELS = 2
CELL_GRID_N = 13
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
FRACTAL_SIZE = CELL_INNER ** FRACTAL_LEVELS
PLACEMENT_RANGE = CELL_GRID_N - FRACTAL_SIZE + 1
COLOR_POOL = 10
MAX_RETRIES = 500

ROLE_COUNT_OPTIONS = ((2, 2), (3, 1), (3, 2))
GIVEN_RECUR_COPY_COUNTS = (1, 2)
GIVEN_SHALLOW_BLOCK_COUNTS = (1,)


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


def _is_visibly_asymmetric(arr: np.ndarray) -> bool:
    """Reject patterns whose typed geometry is unchanged by basic symmetries."""
    return not any(
        np.array_equal(arr, transformed)
        for transformed in (
            np.fliplr(arr),
            np.flipud(arr),
            np.rot90(arr, 1),
            np.rot90(arr, 2),
            np.rot90(arr, 3),
        )
    )


def _expand_mixed_fractal(base_types: np.ndarray) -> np.ndarray:
    """
    Expand a 3-state 3x3 template to 9x9:
    - 0 -> keep majority
    - 1 -> embed the original 3x3 template
    - 2 -> fill a solid 3x3 shallow block
    """
    m = base_types.shape[0]
    out = np.zeros((m * m, m * m), dtype=int)
    for r in range(m):
        for c in range(m):
            rs = slice(r * m, (r + 1) * m)
            cs = slice(c * m, (c + 1) * m)
            if base_types[r, c] == 1:
                out[rs, cs] = base_types
            elif base_types[r, c] == 2:
                out[rs, cs] = 2
    return out


def _pure_recursive_occupied(base_types: np.ndarray) -> np.ndarray:
    """Baseline for degeneration checks: treat every nonzero cell as recursively active."""
    occupied = (base_types > 0).astype(int)
    return np.kron(occupied, occupied).astype(bool)


def _precompute_base_patterns() -> list[np.ndarray]:
    """Enumerate small three-state templates that stay legible after the mixed recursion."""
    templates: list[np.ndarray] = []
    positions = list(range(CELL_INNER * CELL_INNER))
    for recur_count, shallow_count in ROLE_COUNT_OPTIONS:
        for recur_combo in combinations(positions, recur_count):
            recur_set = set(recur_combo)
            remaining = [idx for idx in positions if idx not in recur_set]
            for shallow_combo in combinations(remaining, shallow_count):
                arr = np.zeros((CELL_INNER, CELL_INNER), dtype=int)
                arr.flat[list(recur_combo)] = 1
                arr.flat[list(shallow_combo)] = 2

                occupied = arr > 0
                if not (occupied[0, :].any() and occupied[-1, :].any() and occupied[:, 0].any() and occupied[:, -1].any()):
                    continue
                if np.count_nonzero(arr == 1) < 2 or np.count_nonzero(arr == 2) < 1:
                    continue
                if not _is_visibly_asymmetric(arr):
                    continue
                if not any(
                    np.any(arr[edge] == 2)
                    for edge in ((0, slice(None)), (-1, slice(None)), (slice(None), 0), (slice(None), -1))
                ):
                    continue
                templates.append(arr)
    return templates


_BASE_PATTERNS = _precompute_base_patterns()


def _extract_cell_patterns(g: np.ndarray) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]], int]:
    """Read every cell interior and cache unique patterns."""
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
    return patterns, cell_pat, step


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a two-level fractal in which one blueprint color recurses and another stops after the first expansion.

    Input structure:
    - The canvas is split into equal 3x3 interior cells by full separator rows and columns.
    - Most cells share one background pattern, while two rarer cell motifs also appear:
      a three-color blueprint cell and a solid shallow-color cell.
    - The blueprint cell's own 3x3 pixels form the source template for a 9x9 macro-pattern.

    Rule:
    - Decode the blueprint cell into three roles: shared pixels stay majority, recursive-color pixels expand into another full copy of the 3x3 template, and shallow-color pixels expand only into a solid 3x3 shallow block.
    - This produces a typed 9x9 fractal window on the cell grid.
    - Find the unique placement whose already-shown rare cells exactly agree with that typed window.
    - Fill the whole placed window: majority cells at type 0, blueprint cells at type 1, solid shallow cells at type 2.

    Key insight: The odd cell is not just a binary mask. Its two non-background colors have different recursion depth, so one color keeps reproducing the full template while the other collapses into a one-level solid block.
    """
    g = np.asarray(grid, dtype=int)
    patterns, cell_pat, step = _extract_cell_patterns(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            pat = cell_pat[i][j]
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) == 3, "must have exactly 3 cell patterns (majority + blueprint + shallow)"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    blueprint_candidates = [pat for pat in counts if pat != majority_pat and len(np.unique(patterns[pat])) == 3]
    shallow_candidates = [pat for pat in counts if pat != majority_pat and len(np.unique(patterns[pat])) == 1]
    assert len(blueprint_candidates) == 1, "need a unique 3-color blueprint cell"
    assert len(shallow_candidates) == 1, "need a unique solid shallow cell"

    blueprint_pat = blueprint_candidates[0]
    shallow_pat = shallow_candidates[0]
    blueprint_arr = patterns[blueprint_pat]
    shallow_arr = patterns[shallow_pat]

    blueprint_colors = set(map(int, np.unique(blueprint_arr)))
    majority_colors = set(map(int, np.unique(majority_arr)))
    shared_colors = blueprint_colors & majority_colors
    assert len(shared_colors) == 1, "majority and blueprint must share exactly one color"
    shared_color = next(iter(shared_colors))

    shallow_color = int(shallow_arr[0, 0])
    assert np.all(shallow_arr == shallow_color), "shallow cell must be solid"
    assert shallow_color in blueprint_colors and shallow_color != shared_color, "shallow color must come from the blueprint"

    recur_colors = [color for color in blueprint_colors if color not in (shared_color, shallow_color)]
    assert len(recur_colors) == 1, "blueprint must contain exactly one recursive color"
    recur_color = recur_colors[0]

    base_types = np.zeros_like(blueprint_arr, dtype=int)
    base_types[blueprint_arr == recur_color] = 1
    base_types[blueprint_arr == shallow_color] = 2
    assert np.all((blueprint_arr == shared_color) | (base_types > 0)), "blueprint may only use shared/recur/shallow colors"

    fractal_types = _expand_mixed_fractal(base_types)
    fh, fw = fractal_types.shape

    existing = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if cell_pat[i][j] == blueprint_pat:
                existing[i, j] = 1
            elif cell_pat[i][j] == shallow_pat:
                existing[i, j] = 2
            else:
                assert cell_pat[i][j] == majority_pat, "unexpected cell pattern"

    observed = existing > 0
    candidates: list[tuple[int, int]] = []
    for top in range(0, n - fh + 1):
        for left in range(0, n - fw + 1):
            block = np.zeros((n, n), dtype=bool)
            block[top : top + fh, left : left + fw] = True
            if np.any(observed & ~block):
                continue
            sub_existing = existing[top : top + fh, left : left + fw]
            mask = sub_existing > 0
            if np.any(sub_existing[mask] != fractal_types[mask]):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "must have a unique mixed-fractal placement"
    top, left = candidates[0]

    out = g.copy()
    for i in range(fh):
        for j in range(fw):
            if fractal_types[i, j] == 1:
                target_pat = blueprint_arr
            elif fractal_types[i, j] == 2:
                target_pat = shallow_arr
            else:
                target_pat = majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


def _random_majority_pattern(shared: int, maj_exclusive: int) -> np.ndarray:
    """Generate a non-uniform majority motif using exactly two colors."""
    arr = np.array(
        [random.choice((shared, maj_exclusive)) for _ in range(CELL_INNER * CELL_INNER)],
        dtype=int,
    ).reshape(CELL_INNER, CELL_INNER)
    if np.all(arr == arr[0, 0]):
        arr[random.randrange(CELL_INNER), random.randrange(CELL_INNER)] = maj_exclusive if arr[0, 0] == shared else shared
    if not np.any(arr == shared):
        arr[0, 0] = shared
    if not np.any(arr == maj_exclusive):
        arr[-1, -1] = maj_exclusive
    return arr


def _relative_cells(layout: np.ndarray) -> list[tuple[int, int, int]]:
    """Return non-majority cells as (r, c, type) triples."""
    return [(int(r), int(c), int(layout[r, c])) for r, c in np.argwhere(layout > 0)]


def _recursive_copy_cells(base_types: np.ndarray, macro_pos: tuple[int, int]) -> list[tuple[int, int, int]]:
    """Reveal a full recursive subcopy inside one top-level recursive macro-cell."""
    mr, mc = macro_pos
    cells: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(base_types > 0):
        cells.append((mr * CELL_INNER + int(r), mc * CELL_INNER + int(c), int(base_types[r, c])))
    return cells


def _shallow_block_cells(macro_pos: tuple[int, int]) -> list[tuple[int, int, int]]:
    """Reveal a full 3x3 shallow macro-block for one top-level shallow cell."""
    mr, mc = macro_pos
    return [
        (mr * CELL_INNER + r, mc * CELL_INNER + c, 2)
        for r in range(CELL_INNER)
        for c in range(CELL_INNER)
    ]


def _layout_from_relative_cells(rel_cells: list[tuple[int, int, int]], top: int, left: int) -> np.ndarray:
    """Place a relative typed subset into the 13x13 cell grid."""
    layout = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=int)
    for r, c, cell_type in rel_cells:
        layout[top + r, left + c] = cell_type
    return layout


def _placement_candidates(fractal_types: np.ndarray, observed_layout: np.ndarray) -> list[tuple[int, int]]:
    """Find all placements whose typed 9x9 window agrees with the observed rare cells."""
    fh, fw = fractal_types.shape
    observed = observed_layout > 0
    candidates: list[tuple[int, int]] = []
    for top in range(PLACEMENT_RANGE):
        for left in range(PLACEMENT_RANGE):
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[top : top + fh, left : left + fw] = True
            if np.any(observed & ~block):
                continue
            sub = observed_layout[top : top + fh, left : left + fw]
            mask = sub > 0
            if np.any(sub[mask] != fractal_types[mask]):
                continue
            candidates.append((top, left))
    return candidates


def _choose_given_subset(base_types: np.ndarray, fractal_types: np.ndarray, top: int, left: int) -> list[tuple[int, int, int]] | None:
    """
    Make the fusion visible while keeping the output incomplete:
    - reveal 1-2 full recursive subcopies
    - reveal 1 full shallow 3x3 macro-block
    - optionally add 0-3 extra anchors to force a unique placement
    """
    macro_recur = [tuple(coord) for coord in np.argwhere(base_types == 1)]
    macro_shallow = [tuple(coord) for coord in np.argwhere(base_types == 2)]
    all_rel = _relative_cells(fractal_types)
    all_rel_set = set(all_rel)

    recur_sizes = [size for size in GIVEN_RECUR_COPY_COUNTS if size <= len(macro_recur)]
    shallow_sizes = [size for size in GIVEN_SHALLOW_BLOCK_COUNTS if size <= len(macro_shallow)]
    random.shuffle(recur_sizes)
    random.shuffle(shallow_sizes)

    for recur_size in recur_sizes:
        recur_combos = list(combinations(macro_recur, recur_size))
        random.shuffle(recur_combos)
        for shallow_size in shallow_sizes:
            shallow_combos = list(combinations(macro_shallow, shallow_size))
            random.shuffle(shallow_combos)
            for recur_subset in recur_combos:
                for shallow_subset in shallow_combos:
                    chosen: set[tuple[int, int, int]] = set()
                    for macro_pos in recur_subset:
                        chosen.update(_recursive_copy_cells(base_types, macro_pos))
                    for macro_pos in shallow_subset:
                        chosen.update(_shallow_block_cells(macro_pos))
                    if len(chosen) >= len(all_rel_set):
                        continue

                    remaining = list(all_rel_set - chosen)
                    random.shuffle(remaining)

                    extra_options: list[tuple[tuple[int, int, int], ...]] = [tuple()]
                    extra_options.extend((cell,) for cell in remaining)
                    if len(remaining) >= 2:
                        pair_pool = list(combinations(remaining, 2))
                        random.shuffle(pair_pool)
                        extra_options.extend(pair_pool[:24])
                    if len(remaining) >= 3:
                        triple_pool = list(combinations(remaining, 3))
                        random.shuffle(triple_pool)
                        extra_options.extend(triple_pool[:24])

                    for extras in extra_options:
                        rel_cells = sorted(chosen | set(extras))
                        observed_layout = _layout_from_relative_cells(rel_cells, top, left)
                        if _placement_candidates(fractal_types, observed_layout) == [(top, left)]:
                            return rel_cells
    return None


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(
    separator: int,
    majority: np.ndarray,
    blueprint: np.ndarray,
    shallow: np.ndarray,
    layout: np.ndarray,
) -> np.ndarray:
    """Stamp a full typed cell layout into the pixel grid."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in np.argwhere(layout == 1):
        paste(grid, blueprint, *_cell_origin(int(i), int(j)))
    for i, j in np.argwhere(layout == 2):
        paste(grid, shallow, *_cell_origin(int(i), int(j)))
    return grid


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, recur_color, shallow_color = random.sample(range(COLOR_POOL), 5)
    majority = _random_majority_pattern(shared, maj_exclusive)
    base_types = random.choice(_BASE_PATTERNS).copy()
    blueprint = np.full((CELL_INNER, CELL_INNER), shared, dtype=int)
    blueprint[base_types == 1] = recur_color
    blueprint[base_types == 2] = shallow_color
    shallow = np.full((CELL_INNER, CELL_INNER), shallow_color, dtype=int)

    fractal_types = _expand_mixed_fractal(base_types)
    pure_recursive = _pure_recursive_occupied(base_types)
    fusion_delta = int(np.count_nonzero((fractal_types > 0) != pure_recursive))
    if fusion_delta == 0:
        return None

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)

    out_layout = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=int)
    out_layout[top : top + FRACTAL_SIZE, left : left + FRACTAL_SIZE] = fractal_types
    out = _build_grid(separator, majority, blueprint, shallow, out_layout)

    given_subset = _choose_given_subset(base_types, fractal_types, top, left)
    if given_subset is None:
        return None

    inp_layout = _layout_from_relative_cells(given_subset, top, left)
    inp = _build_grid(separator, majority, blueprint, shallow, inp_layout)

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
    raise RuntimeError("failed to generate a valid mixed-recursive b99e7126 input")
