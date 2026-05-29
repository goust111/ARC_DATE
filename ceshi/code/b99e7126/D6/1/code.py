# Variant of b99e7126
# Dimension: D6 — 网格铺贴结构：单元格的重复模式 (tiling)
# Change: 整个网格由完全相同的NxN单元格无缝平铺，只有一个单元格图案不同... → 行列交替的条带式铺贴

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# Design analysis
# a) Concrete behavior:
#    原题的 7x7 单元格底板只有一种重复单元格；这里改成两种底板单元格 A/B，
#    并且它们按“整行交替”或“整列交替”重复。异常单元格 M 仍然携带 3x3 掩码，
#    但输出时，掩码为 False 的格子不再统一回填同一个 majority，而是回填该位置条带
#    规则下本来应该出现的 A 或 B。一个具体追踪：
#    原题中 block 内 9 个格子里，False 位全部写成 majority；
#    变体中若是 horizontal stripes，则 block 的第 0/2 行应该是 A，第 1 行应该是 B
#    （或反过来），所以 False 位要按行奇偶分别回填，vertical stripes 同理按列奇偶。
# b) Complexity calibration:
#    只保留两种条带模式和两种方向，避免自由度过多；同时强制 3x3 覆盖块里的
#    active/inactive 都跨越两种条带 parity，使视觉上能明显看出“不是单一 majority”。
# c) Broken assumptions from the original solver:
#    - `majority_pat = max(counts, ...)`：原题成立，这里不成立，因为常驻底板有 A/B 两种。
#    - `target_pat = minority if mask else majority`：原题安全，这里会错误地抹掉条带信息。
#    - 其余 placement 搜索逻辑仍成立，因为“已有异常单元格是掩码 active 位的真子集”
#      这个前提没有变，变的只是 active 之外该恢复什么。
# d) Placement legality:
#    生成器仍使用 output-first；先造完整条带底板，再按 mask 覆盖 minority，
#    然后只保留一部分 active 位作为输入。所有单元格都落在固定 7x7 cell grid 内，
#    无重叠、无越界。
# e) Visual observability:
#    解题者需要先看出整行或整列在交替使用 A/B，再看出 M 的内部图案仍然是一个
#    3x3 单元级蓝图。方向在样例间随机为 row/col，两类都由多次重复支撑。
# f) Visual cleanliness:
#    仍然只有 divider + A/B 共用两色 + minority 专属色，颜色数保持 4；A/B 用不同的
#    3x3 纹理而非额外颜色，避免画面过杂。
# g) Color rule:
#    没有按颜色 id 做任何排序或语义推断。active 色仍由 minority 内部“出现最多的颜色”
#    决定，但生成器保证它是唯一多数色，因此这是可见结构而不是数值技巧。
# h) Variant-specific anti-degeneration guard:
#    生成器只接受那些在条带方向上同时覆盖两种 stripe parity 的 mask/placement/subset。
#    因此旧 solver 若把 False 位统一刷成一种 majority，结果通常会和新规则不同。
# i) New solver assumptions and generator guarantees:
#    - 输入中恰有 3 种 cell interior：A、B、M。
#    - 非 M 的格子严格符合“整行交替”或“整列交替”之一。
#    - 已给出的 M 格子数量为 3 或 4，且唯一确定 mask 放置位置。
#    这些都由 `_gen_attempt()` 显式验证，并用 `main(inp) == out` 兜底检查。


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


def _extract_cell_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Read each cell interior into a pattern dictionary and an n x n pattern grid."""
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
    return patterns, cell_pat


def _infer_stripe_tiling(
    cell_pat: list[list[tuple[int, ...]]], minority_pat: tuple[int, ...]
) -> tuple[str, list[list[tuple[int, ...]]]]:
    """
    Recover whether the persistent background alternates by rows or by columns.

    Assumptions:
    - exactly two recurring base patterns remain after removing the minority pattern
    - every non-minority cell follows one alternating stripe rule
    """
    n = len(cell_pat)
    base_patterns = sorted({p for row in cell_pat for p in row if p != minority_pat})
    assert len(base_patterns) == 2, "expected exactly 2 stripe base patterns and 1 minority pattern"

    candidate_grids: dict[tuple[tuple[tuple[int, ...], ...], ...], tuple[str, list[list[tuple[int, ...]]]]] = {}
    for orientation in ("row", "col"):
        for even_pat, odd_pat in (base_patterns, base_patterns[::-1]):
            expected_grid: list[list[tuple[int, ...]]] = []
            ok = True
            for i in range(n):
                row: list[tuple[int, ...]] = []
                for j in range(n):
                    stripe_idx = i if orientation == "row" else j
                    expected = even_pat if stripe_idx % 2 == 0 else odd_pat
                    row.append(expected)
                    if cell_pat[i][j] != minority_pat and cell_pat[i][j] != expected:
                        ok = False
                        break
                if not ok:
                    break
                expected_grid.append(row)
            if ok:
                key = tuple(tuple(row) for row in expected_grid)
                candidate_grids[key] = (orientation, expected_grid)

    assert len(candidate_grids) == 1, "stripe direction or A/B alternation must be uniquely inferable"
    return next(iter(candidate_grids.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the odd cell's 3x3 motif to the cell-grid while preserving alternating stripe tiles.

    Input structure:
    - The board is split into a 7x7 grid of 3x3 cell interiors separated by divider lines.
    - Normal cells do not all match: instead, whole rows or whole columns alternate between two recurring base patterns.
    - A small subset of cells use one distinct minority pattern whose interior still acts as the blueprint.

    Rule:
    - Read the minority cell and turn its dominant color into a boolean 3x3 mask.
    - Find the unique 3x3 placement where the already-shown minority cells fit that mask.
    - Fill every mask-True position with the minority cell, and every mask-False position with the stripe pattern that the row/column alternation demands at that location.

    Key insight:
    - The minority cell still describes a fractal 3x3 region, but the background is no longer uniform.
    - Solving now requires recovering the stripe direction and the A/B row-or-column alternation before completing the scaled-up pattern.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns, cell_pat = _extract_cell_patterns(g, step)

    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) == 3, "variant expects exactly 3 cell patterns: stripe A, stripe B, minority"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    _, stripe_grid = _infer_stripe_tiling(cell_pat, minority_pat)

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat

    candidates: list[tuple[int, int]] = []
    for top in range(n - mh + 1):
        for left in range(n - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))

    assert len(candidates) == 1, "must have a unique mask placement aligning existing minority cells"
    top, left = candidates[0]

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target_pat = minority_arr if mask[i, j] else patterns[stripe_grid[top + i][left + j]]
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (o5x) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 400
ORIENTATIONS = ("row", "col")

# Curated stripe motifs: small, visible, and intentionally distinct while using only 2 colors.
_STRIPE_MASKS = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool),
]

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _precompute_masks():
    """Enumerate valid LR-symmetric minority masks, keyed by active pixel count."""
    weights = [len(group) for group in _LR_GROUPS]
    by_count = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks = []
        for k in range(1, len(_LR_GROUPS) + 1):
            for combo in combinations(range(len(_LR_GROUPS)), k):
                if sum(weights[i] for i in combo) != inactive_count:
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


def _random_minority_mask():
    """Pick a valid minority mask with 6 or 7 active pixels."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _random_stripe_patterns(shared: int, stripe_exclusive: int) -> tuple[np.ndarray, np.ndarray]:
    """Choose two distinct 3x3 base tile patterns for the alternating stripes."""
    idx_a, idx_b = random.sample(range(len(_STRIPE_MASKS)), 2)
    pat_a = np.where(_STRIPE_MASKS[idx_a], stripe_exclusive, shared).astype(int)
    pat_b = np.where(_STRIPE_MASKS[idx_b], stripe_exclusive, shared).astype(int)
    return pat_a, pat_b


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _stripe_parity(rel_cells, top: int, left: int, orientation: str) -> set[int]:
    """Return which row/col parities the given relative coordinates occupy inside the placed block."""
    if orientation == "row":
        return {(top + r) % 2 for r, _ in rel_cells}
    return {(left + c) % 2 for _, c in rel_cells}


def _pattern_grid(stripe_a: np.ndarray, stripe_b: np.ndarray, orientation: str) -> list[list[np.ndarray]]:
    """Build the full 7x7 expected base-tile assignment."""
    grid: list[list[np.ndarray]] = []
    for i in range(CELL_GRID_N):
        row: list[np.ndarray] = []
        for j in range(CELL_GRID_N):
            stripe_idx = i if orientation == "row" else j
            row.append(stripe_a if stripe_idx % 2 == 0 else stripe_b)
        grid.append(row)
    return grid


def _build_grid(separator: int, base_grid: list[list[np.ndarray]], minority: np.ndarray, active_cells) -> np.ndarray:
    """Construct the full pixel grid from the stripe base and an optional set of minority cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, base_grid[i][j], *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _placement_candidates(mask: np.ndarray, subset, top: int, left: int):
    """Find all offsets where the given subset of active cells is consistent with mask."""
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


def _choose_given_subset(mask: np.ndarray, top: int, left: int, orientation: str):
    """Pick 3-4 shown minority cells that still uniquely determine the placement."""
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(active, size)
            if _placement_candidates(mask, subset, top, left) == [(top, left)]
            and len(_stripe_parity(subset, top, left, orientation)) == 2
        ]
        if valid:
            return random.choice(valid)
    return None


def _variant_visible(mask: np.ndarray, top: int, left: int, orientation: str) -> bool:
    """
    Require the 3x3 block to involve both stripe parities in both its active and inactive sets.

    This prevents the task from collapsing back to the single-majority version inside the solved area.
    """
    active = [tuple(coord) for coord in np.argwhere(mask)]
    inactive = [tuple(coord) for coord in np.argwhere(~mask)]
    return len(_stripe_parity(active, top, left, orientation)) == 2 and len(_stripe_parity(inactive, top, left, orientation)) == 2


def _gen_attempt():
    """Single generation attempt: build output, choose input subset, verify with solver."""
    separator, shared, stripe_exclusive, min_exclusive = random.sample(range(COLOR_POOL), 4)
    stripe_a, stripe_b = _random_stripe_patterns(shared, stripe_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)
    orientation = random.choice(ORIENTATIONS)

    placements = [(t, l) for t in range(PLACEMENT_RANGE) for l in range(PLACEMENT_RANGE)]
    random.shuffle(placements)
    chosen = None
    for top, left in placements:
        if not _variant_visible(mask, top, left, orientation):
            continue
        subset = _choose_given_subset(mask, top, left, orientation)
        if subset is not None:
            chosen = (top, left, subset)
            break
    if chosen is None:
        return None

    top, left, subset = chosen
    base_grid = _pattern_grid(stripe_a, stripe_b, orientation)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    given_cells = [(top + r, left + c) for r, c in subset]

    out = _build_grid(separator, base_grid, minority, active_cells)
    inp = _build_grid(separator, base_grid, minority, given_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 D6-variant input")
