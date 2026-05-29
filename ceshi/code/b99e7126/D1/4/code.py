# Variant of b99e7126
# Dimension: D1 — 匹配条件：异常单元格的识别方式 (match_cond)
# Change: 通过比较所有单元格的内部颜色图案，找到与多数单元格图案不同的那个唯一异常单元格... → 形状匹配：单元格内部图案的几何形状特征
#
# Design analysis
# spec:
#   variant_core:
#     original_match: 单元格按完整 3x3 像素图案计数；出现次数最少的精确图案就是 source。
#     new_match: 单元格先压缩成“主色连通块”的几何形状，再按形状类计数；旋转/翻转与背景装饰都忽略。
#     preserved_rule: source 单元格内部 3x3 形状仍然充当宏观 7x7 单元格网格上的蓝图，输出仍是把缺失的 source 单元格补齐。
#   concrete_pixel_behavior:
#     cell_interior: 每个 3x3 单元格都有一个占 6-7 像素的主色连通块；其余 2-3 个像素是装饰背景。
#     majority_class: 多数单元格都属于同一个几何形状类，但会随机旋转/翻转，并且背景 2-3 个像素的配色也会变化，所以像素级图案高度多样。
#     source_class: 少数单元格共享另一种几何形状类，且保持同一朝向；它们的主色掩码就是要在 7x7 单元格网格上放大的分形模板。
#     example_trace:
#       original_solver_step: 如果某个 majority 形状分别以 30 多种像素图案出现，而 source 精确图案只出现 4 次，旧逻辑仍会去找“最少的精确图案”，反而拿到只出现 1 次的 majority 装饰变体。
#       variant_solver_step: 先把每个单元格还原成主色布尔掩码，再对掩码做 D4 规范化；30 多种 majority 像素图案都会并到同一个形状类，source 的 4 个单元格会并到另一个形状类。
#       key_difference: 旧逻辑比较的是像素阵列；新逻辑比较的是去颜色装饰后的几何类。
#   complexity_calibration:
#     support_repetition: 多数类在 49 个单元格里重复几十次，少数类重复 3-5 次，样例中有足够支撑让人看到“这一类形状才是线索”。
#     anti_degeneration: generator 刻意制造大量 exact-pattern 唯一的 majority 单元格，避免退化回“像素计数也能解”。
#   old_assumption_audit:
#     - old_main 用 exact pattern 计数找 minority；在本变体中不成立，因为多数类被拆成很多像素变体。
#     - old_main 假设 block 内 inactive 位置都应写回同一个 majority 图案；在本变体中不成立，因为 majority 本来就允许同形异纹，所以输出只能补 source，不能统一重绘 majority。
#     - old_main 用 minority 的最常见颜色当 active color；这里保留，但 generator 保证每个单元格主色都唯一最多且 4 连通。
#   placement_legality:
#     - 仍使用 7x7 单元格网格、3x3 内部、1 像素分隔线，source 蓝图仍是 3x3。
#     - source 掩码固定为连通且可放入 7x7 的形状；给定的 source 子集必须唯一确定其 3x3 宏观放置偏移。
#     - 先构造“全 majority”的底板，再覆盖全部 source 得到 output，只覆盖部分 source 得到 input，因此不会越界或出现非法重叠。
#   visual_observability:
#     - 所有单元格共享同一个显眼的主色连通块，解题者能肉眼看到“绝大多数是同一种几何形状，只是朝向和背景点缀不同”。
#     - source 与 majority 使用同一主色，避免把颜色 id 当成真正线索；差异主要体现在形状上。
#   neatness:
#     - 装饰只存在于每个 3x3 单元格内部的 2-3 个剩余像素，不会与宏观覆盖层叠在一起造成混乱。
#   color_reasoning:
#     - 总颜色数控制在 5 个：分隔线色、主色、两种常规背景色、1 种额外装饰色。
#     - 颜色变化只服务于“打散像素图案”，真正决定类别的是主色连通形状。
#   new_solver_assumptions:
#     - 每个单元格都存在唯一最多的颜色，且该颜色形成单个 4 连通块。
#     - 规范化后的形状类只有两个：majority 类与 source 类。
#     - source 类在输入中至少出现 3 次，足以唯一定位 3x3 宏观模板；generator 会显式检查这一点。

from functools import reduce
from itertools import combinations, product
from math import gcd

import numpy as np
import random
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the fractal overlay by finding the rare cell *shape class* instead of the rare exact bitmap.

    Input structure:
    - A 7x7 cell grid is separated by single-color divider lines, with 3x3 interiors inside each cell.
    - Every cell contains one dominant connected shape color plus a few decorative background pixels.
    - Most cells share one geometric shape class, but their exact 3x3 pixels can differ by rotation, reflection, and decoration.
    - A few cells share a second shape class; these are the source clues.

    Rule:
    - Reduce each cell to the mask of its dominant connected color and group those masks by shape class up to rotation/flip.
    - Treat the rarer shape class as the source class and take one exemplar cell from it as the stamp.
    - Use that exemplar's dominant-color mask as the 3x3 macro blueprint, locate its unique placement on the 7x7 cell grid, and fill every active macro position with the exemplar cell.

    Key insight: The important anomaly is geometric, not pixel-exact — many majority cells look different at the bitmap level but collapse to the same canonical shape once decorative colors and orientation are ignored.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    cell_arrs: list[list[np.ndarray]] = []
    shape_sig: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row_arrs = []
        row_sig = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            _, mask = _dominant_shape(arr)
            row_arrs.append(arr)
            row_sig.append(_canonical_shape(mask))
        cell_arrs.append(row_arrs)
        shape_sig.append(row_sig)

    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            sig = shape_sig[i][j]
            counts[sig] = counts.get(sig, 0) + 1
    assert len(counts) == 2, "expected exactly two shape classes"

    source_sig = min(counts.items(), key=lambda kv: kv[1])[0]

    source_arr = None
    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            is_source = shape_sig[i][j] == source_sig
            existing[i, j] = is_source
            if is_source and source_arr is None:
                source_arr = cell_arrs[i][j]
    assert source_arr is not None, "need at least one source exemplar"

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
    assert len(candidates) == 1, "must have a unique placement for the source shape mask"
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


# --------------- generator (o5x) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
GIVEN_CELL_SIZES = (3, 4, 5)
COLOR_POOL = 10
MAX_RETRIES = 300

SOURCE_MASK = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
    ],
    dtype=bool,
)

MAJORITY_MASK = np.array(
    [
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
    ],
    dtype=bool,
)

assert _canonical_shape(SOURCE_MASK) != _canonical_shape(MAJORITY_MASK)
MAJORITY_MASK_VARIANTS = _symmetry_variants(MAJORITY_MASK)


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_source_pattern(shape_color: int, bg_a: int, bg_b: int) -> np.ndarray:
    """Fixed source tile whose dominant-color mask is the macro blueprint."""
    arr = np.full((MASK_SIZE, MASK_SIZE), bg_a, dtype=int)
    arr[SOURCE_MASK] = shape_color
    arr[0, 0] = bg_b
    arr[0, 2] = bg_a
    return arr


def _majority_variants(shape_color: int, bg_a: int, bg_b: int, accent: int) -> list[np.ndarray]:
    """Many exact majority bitmaps sharing one canonical shape class."""
    fills = (bg_a, bg_b, accent)
    variants = []
    seen = set()
    for mask in MAJORITY_MASK_VARIANTS:
        gaps = [tuple(int(v) for v in rc) for rc in np.argwhere(~mask)]
        for assignment in product(fills, repeat=len(gaps)):
            if len(set(assignment)) == 1:
                continue
            arr = np.full((MASK_SIZE, MASK_SIZE), shape_color, dtype=int)
            for (r, c), color in zip(gaps, assignment):
                arr[r, c] = int(color)
            key = tuple(int(v) for v in arr.ravel())
            if key not in seen:
                seen.add(key)
                variants.append(arr)
    assert variants, "need majority variants"
    return variants


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


def _build_base_grid(separator: int, cell_patterns: list[list[np.ndarray]]) -> np.ndarray:
    """Construct the full grid from already chosen per-cell 3x3 interiors."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, cell_patterns[i][j], *_cell_origin(i, j))
    return grid


def _extract_shape_counts(grid: np.ndarray) -> dict[tuple[int, ...], int]:
    """Small helper for generator-side diagnostics and rejection."""
    step, _, _ = _infer_cell_grid_params(grid)
    n = (grid.shape[0] - 1) // step
    inner = step - 1
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = grid[r0 : r0 + inner, c0 : c0 + inner]
            _, mask = _dominant_shape(arr)
            sig = _canonical_shape(mask)
            counts[sig] = counts.get(sig, 0) + 1
    return counts


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: construct output first, then hide some source cells."""
    separator, shape_color, bg_a, bg_b, accent = random.sample(range(COLOR_POOL), 5)
    source = _build_source_pattern(shape_color, bg_a, bg_b)
    majority_bank = _majority_variants(shape_color, bg_a, bg_b, accent)

    total_cells = CELL_GRID_N * CELL_GRID_N
    if len(majority_bank) >= total_cells:
        chosen = random.sample(majority_bank, total_cells)
    else:
        chosen = [random.choice(majority_bank) for _ in range(total_cells)]

    cell_patterns: list[list[np.ndarray]] = []
    idx = 0
    for _ in range(CELL_GRID_N):
        row = []
        for _ in range(CELL_GRID_N):
            row.append(chosen[idx].copy())
            idx += 1
        cell_patterns.append(row)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(SOURCE_MASK)]

    base = _build_base_grid(separator, cell_patterns)
    out = base.copy()
    for i, j in active_cells:
        paste(out, source, *_cell_origin(i, j))

    subset = _choose_given_subset(SOURCE_MASK, top, left)
    if subset is None:
        return None

    inp = base.copy()
    for r, c in subset:
        paste(inp, source, *_cell_origin(top + r, left + c))

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    shape_counts = _extract_shape_counts(inp)
    if sorted(shape_counts.values()) != [len(subset), total_cells - len(subset)]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 D1 shape-match input")
