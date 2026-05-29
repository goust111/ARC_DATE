# Combo variant of b99e7126 (index 32)
# combo_spec: D1/4+D12/2
# Name: 内部对称性 + 形状匹配
#
# Design analysis
# spec:
#   variant_core:
#     original_match: 原题把 3x3 单元格按精确像素图案计数，次数最少的 exact pattern 就是 source。
#     fused_match: 先观察“多数单元格整体都左右对称”，把这些对称单元格视为正常背景家族；再只在少数非对称单元格里做主色连通形状匹配，恢复 source 类。
#     preserved_rule: source 单元格的主色掩码仍然是 3x3 宏观蓝图；输出仍是在 7x7 单元格网格上把缺失的 source 补齐。
#   concrete_pixel_behavior:
#     cell_interior: 每个 3x3 单元格都有一个唯一最多、且 4 连通的主色块，面积为 5-7 像素。
#     majority_family: 所有正常单元格整个 3x3 阵列都满足左右镜像对称，但它们并不只有一个形状类；生成器会混入 5 个左右对称的主色形状类，并在空隙位置换不同背景配色。
#     source_family: source 单元格整体故意打破左右对称，且共享同一个非对称主色形状；这个形状既是“异常类别”的识别依据，也是宏观补全蓝图。
#     example_trace:
#       original_solver_step: 如果输入里某个对称 majority 图案只出现 1 次，而 source 精确图案出现 3 次，旧逻辑会误把那块 1 次的对称 majority 当成 minority。
#       shape_only_step: 如果只按主色形状全局计数，生成器还会安排 1-2 个“很少见但仍然对称”的 majority 形状类，因此“全局最少形状类”也未必是 source。
#       fused_step: 先用左右对称性过滤出正常背景家族，再在非对称单元格里按主色连通形状做匹配，就能稳定锁定 source；随后把该 source 的主色掩码放大到 7x7 单元格层级完成补全。
#       key_difference: 新规则把“正常图案的内部对称性”和“异常图案的几何形状类别”绑在同一条推理链上，而不是单独依赖颜色 id 或 exact pattern 频次。
#   complexity_calibration:
#     support_repetition: source 在输入中显示 3-4 次，足以看出“这些非对称单元格是一类”；majority 至少含 5 个左右对称形状类，其中 2 类只有 1-2 次，用来破坏“最少形状=source”的偷懒解法。
#     anti_degeneration: 生成器显式拒绝两种退化样本：一是 source 变成全局最少的形状类，二是某个多数单元格失去左右对称性。
#   old_assumption_audit:
#     - old_main 假设“多数类只有一个 exact pattern”；这里不成立，因为多数背景允许多个对称形状类和大量装饰变体。
#     - old_main 假设“最少出现的精确图案就是 source”；这里不成立，因为 generator 刻意制造只出现 1 次的对称 majority。
#     - old_main 假设 block 内 inactive 位置都该重绘成同一种 majority 图案；这里也不成立，因为正常背景本来就允许同属对称家族但纹理不同，所以输出只能补 source，不能统一重刷 majority。
#     - old_main 用 minority 图案本身定义 source；这里必须先确认“哪些单元格打破了共同对称性”，否则 shape count 会被稀有的对称 majority 干扰。
#   placement_legality:
#     - 仍使用 7x7 单元格网格、3x3 内部、1 像素分隔线，source 蓝图仍是 3x3。
#     - source 掩码只从连通、非左右对称、且四边都有覆盖的 3x3 掩码中选取，保证宏观放置信息足够强。
#     - 先构造全对称 majority 底板，再把 source 覆盖到宏观蓝图激活的位置，input 只保留其中 3-4 个 source，因此不会重叠或越界。
#   visual_observability:
#     - 人眼先能看到“绝大多数小格子都是左右对称的”，再注意到少数非对称格子共享同一种主色几何轮廓。
#     - source 与 majority 共享同一个主色，避免把颜色本身误读为规则；真正的线索是“对称 vs 非对称”和“主色块形状”。
#   neatness:
#     - 变化都限制在各自 3x3 单元格内部，宏观层面只是把若干单元格替换成 source，不会出现多层覆盖互相遮挡。
#   color_reasoning:
#     - 总颜色数控制在 5 个：分隔线色、主色、两种常规背景色、1 种装饰色。
#     - 背景颜色只负责打散 exact pattern，并不参与排序；任何判断都基于可见的对称性和连通形状。
#   new_solver_assumptions:
#     - 所有正常单元格都满足左右镜像对称，所有 source 单元格都不满足。
#     - 每个单元格都存在唯一最多的颜色，且该颜色形成单个 4 连通块。
#     - 非对称单元格在输入中只对应一个 source 形状类，且出现 3-4 次；generator 会验证这一点。

from functools import reduce
from itertools import combinations, product
from math import gcd

import numpy as np
import random
from arc_utils import paste


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
GIVEN_CELL_SIZES = (3, 4)
COLOR_POOL = 10
MAX_RETRIES = 400
MAJORITY_CLASS_COUNT = 5


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


def _symmetry_variants(mask: np.ndarray) -> list[np.ndarray]:
    """Unique rotations/reflections of a 2D boolean mask."""
    variants = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for cand in (rot, np.fliplr(rot)):
            key = tuple(int(v) for v in cand.ravel())
            if key not in seen:
                seen.add(key)
                variants.append(cand.copy())
    return variants


def _canonical_shape(mask: np.ndarray) -> tuple[int, ...]:
    """Canonical D4-invariant signature for a 3x3 shape."""
    return min(tuple(int(v) for v in cand.ravel()) for cand in _symmetry_variants(mask))


def _is_connected(mask: np.ndarray) -> bool:
    """4-connectivity for a boolean mask."""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _dominant_shape(arr: np.ndarray) -> tuple[int, np.ndarray]:
    """Extract the unique most frequent color and its connected mask."""
    vals, cnts = np.unique(arr, return_counts=True)
    order = np.argsort(-cnts)
    assert len(order) >= 2 and cnts[order[0]] > cnts[order[1]], "dominant color must be unique"
    color = int(vals[order[0]])
    mask = arr == color
    assert _is_connected(mask), "dominant color must form one connected shape"
    return color, mask


def _is_lr_symmetric(arr: np.ndarray) -> bool:
    """Whole-cell left/right mirror symmetry."""
    return bool(np.array_equal(arr, np.fliplr(arr)))


def _spans_all_edges(mask: np.ndarray) -> bool:
    """Whether a 3x3 source mask touches top/bottom/left/right edges."""
    return bool(mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any())


def _mirror_groups(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    """Mirror-linked inactive pixel groups for LR-symmetric pattern decoration."""
    seen = set()
    groups = []
    for r, c in np.argwhere(~mask):
        rc = (int(r), int(c))
        if rc in seen:
            continue
        mate = (int(r), int(mask.shape[1] - 1 - c))
        group = tuple(sorted({rc, mate}))
        seen.update(group)
        groups.append(group)
    groups.sort()
    return groups


def _precompute_mask_families() -> tuple[dict[tuple[int, ...], list[np.ndarray]], list[np.ndarray]]:
    """Enumerate LR-symmetric majority masks and non-symmetric source masks."""
    majority_by_sig: dict[tuple[int, ...], list[np.ndarray]] = {}
    source_candidates: list[np.ndarray] = []

    connected_masks: list[np.ndarray] = []
    for bits in product((0, 1), repeat=MASK_SIZE * MASK_SIZE):
        mask = np.array(bits, dtype=bool).reshape(MASK_SIZE, MASK_SIZE)
        active = int(mask.sum())
        if active < 5 or active > 7 or not _is_connected(mask):
            continue
        connected_masks.append(mask)

    majority_sigs = set()
    for mask in connected_masks:
        if _is_lr_symmetric(mask):
            sig = _canonical_shape(mask)
            majority_sigs.add(sig)
            majority_by_sig.setdefault(sig, []).append(mask.copy())

    for mask in connected_masks:
        if _is_lr_symmetric(mask):
            continue
        if not _spans_all_edges(mask):
            continue
        if _canonical_shape(mask) in majority_sigs:
            continue
        source_candidates.append(mask.copy())

    assert len(majority_by_sig) >= MAJORITY_CLASS_COUNT, "need enough majority shape classes"
    assert source_candidates, "need non-symmetric source masks"
    return majority_by_sig, source_candidates


MAJORITY_MASKS_BY_SIG, SOURCE_MASKS = _precompute_mask_families()


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_majority_bank(masks: list[np.ndarray], shape_color: int, bg_colors: tuple[int, int, int]) -> list[np.ndarray]:
    """All LR-symmetric exact patterns for one majority shape class."""
    variants: list[np.ndarray] = []
    seen = set()
    for mask in masks:
        groups = _mirror_groups(mask)
        for assignment in product(bg_colors, repeat=len(groups)):
            if len(groups) > 1 and len(set(assignment)) == 1:
                continue
            arr = np.full((MASK_SIZE, MASK_SIZE), shape_color, dtype=int)
            for group, color in zip(groups, assignment):
                for r, c in group:
                    arr[r, c] = int(color)
            key = tuple(int(v) for v in arr.ravel())
            if key not in seen:
                seen.add(key)
                variants.append(arr.copy())
    assert variants, "need majority variants"
    return variants


def _build_source_pattern(mask: np.ndarray, shape_color: int, bg_a: int, bg_b: int, accent: int) -> np.ndarray:
    """A fixed non-symmetric source tile whose dominant mask is the macro blueprint."""
    arr = np.full((MASK_SIZE, MASK_SIZE), bg_a, dtype=int)
    arr[mask] = shape_color
    gaps = [tuple(int(v) for v in rc) for rc in np.argwhere(~mask)]
    choices = [random.choice((bg_a, bg_b, accent)) for _ in gaps]
    if len(gaps) > 1 and len(set(choices)) == 1:
        choices[0] = bg_b if choices[0] != bg_b else accent
    for (r, c), color in zip(gaps, choices):
        arr[r, c] = int(color)
    assert not _is_lr_symmetric(arr), "source must break LR symmetry"
    return arr


def _choose_variants(bank: list[np.ndarray], count: int) -> list[np.ndarray]:
    """Sample exact majority variants, preferring unique patterns when possible."""
    if count <= len(bank):
        return [arr.copy() for arr in random.sample(bank, count)]
    chosen = [arr.copy() for arr in bank]
    chosen.extend(random.choice(bank).copy() for _ in range(count - len(bank)))
    return chosen


def _allocate_majority_counts(total_majority: int) -> list[int]:
    """Distribute majority cells across 5 symmetric shape classes with a few tiny classes."""
    counts = [1, 2, 5, 5, 5]
    assert len(counts) == MAJORITY_CLASS_COUNT and sum(counts) <= total_majority
    remaining = total_majority - sum(counts)
    while remaining > 0:
        counts[random.randrange(MAJORITY_CLASS_COUNT)] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _build_grid(separator: int, cell_patterns: list[list[np.ndarray]]) -> np.ndarray:
    """Construct the full grid from already chosen per-cell 3x3 interiors."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, cell_patterns[i][j], *_cell_origin(i, j))
    return grid


def _placement_candidates(mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    """Find all offsets where the given subset of active cells fits the source mask."""
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


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...] | None:
    """Choose a source-cell subset that uniquely determines the macro placement."""
    active = [tuple(int(v) for v in coord) for coord in np.argwhere(mask)]
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


def _extract_cell_data(grid: np.ndarray) -> dict[str, object]:
    """Shared parser for solver logic and generator-side diagnostics."""
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    cell_arrs: list[list[np.ndarray]] = []
    shape_sig: list[list[tuple[int, ...]]] = []
    lr_sym = np.zeros((n, n), dtype=bool)
    exact_counts: dict[tuple[int, ...], int] = {}
    shape_counts: dict[tuple[int, ...], int] = {}
    symmetric_shape_counts: dict[tuple[int, ...], int] = {}
    asymmetric_shape_counts: dict[tuple[int, ...], int] = {}

    for i in range(n):
        row_arrs = []
        row_sig = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            _, mask = _dominant_shape(arr)
            sig = _canonical_shape(mask)
            sym = _is_lr_symmetric(arr)
            exact = tuple(int(v) for v in arr.ravel())

            row_arrs.append(arr)
            row_sig.append(sig)
            lr_sym[i, j] = sym
            exact_counts[exact] = exact_counts.get(exact, 0) + 1
            shape_counts[sig] = shape_counts.get(sig, 0) + 1
            target = symmetric_shape_counts if sym else asymmetric_shape_counts
            target[sig] = target.get(sig, 0) + 1
        cell_arrs.append(row_arrs)
        shape_sig.append(row_sig)

    return {
        "grid": g,
        "step": step,
        "n": n,
        "inner": inner,
        "cell_arrs": cell_arrs,
        "shape_sig": shape_sig,
        "lr_sym": lr_sym,
        "exact_counts": exact_counts,
        "shape_counts": shape_counts,
        "symmetric_shape_counts": symmetric_shape_counts,
        "asymmetric_shape_counts": asymmetric_shape_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the fractal overlay by combining a symmetry cue with shape-based source matching.

    Input structure:
    - The 29x29 board is divided into a 7x7 grid of cells, each with a 3x3 interior and 1-pixel separators.
    - Every cell has one dominant connected color block, but the normal background cells are all left-right symmetric as full 3x3 patterns.
    - The background is not a single repeated tile: several different symmetric shape classes and many symmetric decorations are mixed together.
    - A few clue cells break that shared symmetry; those asymmetric cells all belong to one source shape class.

    Rule:
    - Use whole-cell left-right symmetry to separate the normal background family from the asymmetric clue cells.
    - Inside the asymmetric cells, match the dominant connected-color shape to identify the shared source class and take one exemplar tile from it.
    - Treat that exemplar's dominant-color mask as a 3x3 macro blueprint, find the unique placement consistent with the visible source cells, and paste the exemplar tile into every active macro position.

    Key insight: The anomaly is not “the rarest bitmap” and not even “the globally rarest shape” — normal tiles advertise themselves by a common internal symmetry, and only after filtering by that symmetry does the shared non-symmetric source shape emerge as the fractal template.
    """
    data = _extract_cell_data(grid)
    g = data["grid"]
    step = int(data["step"])
    n = int(data["n"])
    inner = int(data["inner"])
    cell_arrs = data["cell_arrs"]
    shape_sig = data["shape_sig"]
    lr_sym = data["lr_sym"]
    asymmetric_shape_counts = data["asymmetric_shape_counts"]

    assert int(lr_sym.sum()) > (n * n) // 2, "symmetric background must be the majority"
    assert np.any(~lr_sym), "need some asymmetric source clues"
    assert len(asymmetric_shape_counts) == 1, "asymmetric clue cells should share one shape class"

    source_sig = next(iter(asymmetric_shape_counts))
    source_arr = None
    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            is_source = (not lr_sym[i, j]) and shape_sig[i][j] == source_sig
            existing[i, j] = is_source
            if is_source and source_arr is None:
                source_arr = cell_arrs[i][j]
    assert source_arr is not None, "need a source exemplar"

    _, mask = _dominant_shape(source_arr)
    mh, mw = mask.shape

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
    assert len(candidates) == 1, "must have a unique placement for the source mask"
    top, left = candidates[0]

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            if not mask[i, j]:
                continue
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = source_arr
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: make symmetric background, then reveal a few asymmetric source clues."""
    separator, shape_color, bg_a, bg_b, accent = random.sample(range(COLOR_POOL), 5)
    bg_colors = (bg_a, bg_b, accent)

    majority_items = random.sample(list(MAJORITY_MASKS_BY_SIG.items()), MAJORITY_CLASS_COUNT)
    source_mask = random.choice(SOURCE_MASKS)
    source_sig = _canonical_shape(source_mask)
    assert all(sig != source_sig for sig, _ in majority_items)
    source = _build_source_pattern(source_mask, shape_color, bg_a, bg_b, accent)
    source_exact = tuple(int(v) for v in source.ravel())

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(source_mask)]
    subset = _choose_given_subset(source_mask, top, left)
    if subset is None:
        return None
    shown_source_cells = {(top + r, left + c) for r, c in subset}

    total_majority = CELL_GRID_N * CELL_GRID_N - len(shown_source_cells)
    shape_counts = _allocate_majority_counts(total_majority)
    majority_patterns: list[np.ndarray] = []
    for (sig, masks), count in zip(majority_items, shape_counts):
        bank = _build_majority_bank(masks, shape_color, bg_colors)
        majority_patterns.extend(_choose_variants(bank, count))
    random.shuffle(majority_patterns)
    assert len(majority_patterns) == total_majority

    cell_patterns: list[list[np.ndarray]] = []
    idx = 0
    for i in range(CELL_GRID_N):
        row: list[np.ndarray] = []
        for j in range(CELL_GRID_N):
            if (i, j) in shown_source_cells:
                row.append(source.copy())
            else:
                row.append(majority_patterns[idx].copy())
                idx += 1
        cell_patterns.append(row)
    assert idx == total_majority

    inp = _build_grid(separator, cell_patterns)

    out = inp.copy()
    for i, j in active_cells:
        paste(out, source, *_cell_origin(i, j))

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    diag = _extract_cell_data(inp)
    lr_sym = diag["lr_sym"]
    asymmetric_shape_counts = diag["asymmetric_shape_counts"]
    symmetric_shape_counts = diag["symmetric_shape_counts"]
    exact_counts = diag["exact_counts"]
    shape_counts_all = diag["shape_counts"]

    if int(lr_sym.sum()) != total_majority:
        return None
    if any(bool(lr_sym[r, c]) for r, c in shown_source_cells):
        return None
    if len(asymmetric_shape_counts) != 1 or next(iter(asymmetric_shape_counts)) != source_sig:
        return None
    if asymmetric_shape_counts[source_sig] != len(shown_source_cells):
        return None
    if min(symmetric_shape_counts.values()) >= len(shown_source_cells):
        return None
    if shape_counts_all.get(source_sig, 0) <= min(shape_counts_all.values()):
        return None
    if exact_counts.get(source_exact, 0) != len(shown_source_cells):
        return None
    if min(exact_counts.values()) >= len(shown_source_cells):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
