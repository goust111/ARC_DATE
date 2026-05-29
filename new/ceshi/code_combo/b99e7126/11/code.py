# Combo variant of b99e7126 (index 11)
# combo_spec: D6/1+D8/0
# Name: 条带式铺贴 + 可变单元格尺寸

import random
from itertools import combinations

import numpy as np

from arc_utils import paste


# Design analysis
# a) Concrete behavior:
#    原题把整个粗网格都铺成同一种底板 cell；这里只保留“异常 cell 内部图案会按 cell 级别放大”
#    这个分形核心，但底板改成条带式：要么整行按 A/B/A/B... 交替，要么整列交替。
#    同时 cell 不再固定 3x3，而是在不同样例里改成 4x4 或 5x5。一个具体追踪：
#    - 原题在 3x3 block 的 mask=False 位置统一回填同一个 majority；
#    - 本融合版先从分隔线推断每个 cell 是 4x4 还是 5x5，再读出 minority cell 的同尺寸 mask；
#    - 找到该 mask 在 8x8 coarse grid 上的唯一放置后，mask=True 位置写回 minority，
#      mask=False 位置则按当前位置所属的条带奇偶写回 A 或 B。
#    差异不只是一项一项叠加：cell 尺寸决定了 mask 的边长与可放置范围，而条带方向决定了
#    mask=False 位置究竟该恢复哪一种底板纹理。
# b) Complexity calibration:
#    只允许两种底板图案 A/B，且条带方向只在 row/col 两档中选，避免自由度过多。
#    另一方面，cell 尺寸固定从少量档位 {4,5} 中采样，并强制 mask 的 active/inactive
#    都跨越行奇偶和列奇偶，这样视觉上既能看出“尺寸变了”，也能看出“背景不是单一 majority”。
# c) Broken assumptions from the original solver:
#    - “最大频次 pattern 就是唯一 majority” 不再成立，因为底板长期存在 A/B 两种 pattern。
#    - “minority 的内部一定是 3x3” 不再成立，必须从 divider spacing 推断真实边长。
#    - “mask=False 位置统一刷成 majority” 不再成立，必须先恢复条带方向与 A/B 对应关系。
#    - placement 搜索本身仍然可用，因为输入依旧只展示 active cells 的严格子集。
# d) Placement legality:
#    生成器继续采用 output-first。先生成完整条带底板，再按 mask 覆盖 minority 得到输出，
#    然后只保留一部分 active positions 作为输入。cell grid 固定为 8x8，mask 尺寸等于 cell
#    边长，所以放置范围始终是 `0..N-inner`，所有覆盖都在 coarse grid 内且不会重叠越界。
# e) Visual observability:
#    人类需要同时看出两件事：一是 divider 之间的 cell 大小在不同样例里不同；二是非异常格并非
#    单一底板，而是整行或整列反复交替 A/B。异常 cell 内部的 4x4 或 5x5 图案仍然直接对应
#    粗网格上的一个同尺寸覆盖块，因此从 2-4 组样例中可肉眼归纳。
# f) Visual cleanliness:
#    仍然只用 4 种颜色：divider、shared、stripe-exclusive、minority-exclusive。A/B 用相同
#    的两色但不同纹理，minority 用 shared + minority-exclusive，从而既能区分三类 cell，
#    又不会因为额外颜色过多而杂乱。
# g) Color rule:
#    solver 不依赖颜色 id 大小。minority 的 active 色通过“内部出现次数最多的颜色”得到；
#    生成器保证该颜色唯一占多数，因此这是可见频率结构，不是数值排序技巧。
# h) Anti-degeneration guard:
#    生成器只接受那些满足以下条件的样例：
#    - mask 的 active 与 inactive 都覆盖两种 stripe parity；
#    - 解出的覆盖块里，mask=False 位置确实同时需要 A 和 B 两种底板；
#    - 选出的已给 minority 子集既唯一确定放置，又跨越两种 stripe parity。
#    因此如果退回原题式“统一刷一个 majority”，会在 solved block 内明显出错。
# i) Solver assumptions vs generator guarantees:
#    - solver 假设输入恰有 3 种 cell interior：A、B、M；generator 显式检查这一点。
#    - solver 假设非 M cells 完全符合某一种行条带或列条带；generator 用同一规则构造底板，
#      再用 `main(inp) == out` 回验。
#    - solver 假设 minority 的可见子集唯一确定 placement；generator 在采样子集时显式检查。


SIZE_CONFIGS = (
    {"inner": 4, "n_cells": 8, "active_counts": (10, 11), "given_sizes": (4, 5, 6)},
    {"inner": 5, "n_cells": 8, "active_counts": (14, 15, 16), "given_sizes": (5, 6, 7)},
)
ORIENTATIONS = ("row", "col")
COLOR_POOL = 10
MAX_RETRIES = 500
SUBSET_SAMPLE_BUDGET = 180


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    """
    Infer a square cell grid separated by full rows/cols of one divider color.
    Returns (step, n_cells, row_divs, col_divs).
    """
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need at least 2 row and 2 col dividers"
    row_divs.sort()
    col_divs.sort()
    assert row_divs[0] == 0 and col_divs[0] == 0, "first divider must be at index 0"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "last divider must be on the boundary"

    row_steps = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1)]
    col_steps = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1)]
    assert len(set(row_steps)) == 1 and len(set(col_steps)) == 1, "divider spacing must be regular"
    step_r = int(row_steps[0])
    step_c = int(col_steps[0])
    assert step_r == step_c, "fusion variant expects square cells"
    step = step_r
    assert step > 1, "cell step must exceed 1"
    assert (h - 1) % step == 0 and (w - 1) % step == 0, "grid dimensions must be k*step+1"
    n_rows = (h - 1) // step
    n_cols = (w - 1) // step
    assert n_rows == n_cols, "fusion variant expects a square coarse grid"
    return step, int(n_rows), row_divs, col_divs


def _extract_cell_patterns(
    g: np.ndarray, step: int, n_cells: int
) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Extract each cell interior both as a hashable key and as its 2D array."""
    inner = step - 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n_cells):
        row: list[tuple[int, ...]] = []
        for j in range(n_cells):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            key = tuple(int(v) for v in arr.ravel())
            if key not in patterns:
                patterns[key] = arr
            row.append(key)
        cell_pat.append(row)
    return patterns, cell_pat


def _infer_stripe_tiling(
    cell_pat: list[list[tuple[int, ...]]], minority_pat: tuple[int, ...]
) -> tuple[str, list[list[tuple[int, ...]]]]:
    """
    Recover whether the persistent background alternates by rows or by columns.
    Returns (orientation, expected_pattern_grid).
    """
    n = len(cell_pat)
    base_patterns = list({pat for row in cell_pat for pat in row if pat != minority_pat})
    assert len(base_patterns) == 2, "expected exactly 2 stripe base patterns and 1 minority pattern"

    candidates: dict[tuple[tuple[tuple[int, ...], ...], ...], tuple[str, list[list[tuple[int, ...]]]]] = {}
    for orientation in ORIENTATIONS:
        for even_pat, odd_pat in (tuple(base_patterns), tuple(base_patterns[::-1])):
            expected_grid: list[list[tuple[int, ...]]] = []
            ok = True
            for i in range(n):
                row: list[tuple[int, ...]] = []
                for j in range(n):
                    stripe_idx = i if orientation == "row" else j
                    expected = even_pat if stripe_idx % 2 == 0 else odd_pat
                    row.append(expected)
                    actual = cell_pat[i][j]
                    if actual != minority_pat and actual != expected:
                        ok = False
                        break
                if not ok:
                    break
                expected_grid.append(row)
            if ok:
                key = tuple(tuple(row) for row in expected_grid)
                candidates[key] = (orientation, expected_grid)

    assert len(candidates) == 1, "stripe direction or parity assignment must be uniquely inferable"
    return next(iter(candidates.values()))


def _find_mask_placement(existing: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """Find the unique placement consistent with the visible minority cells."""
    n_rows, n_cols = existing.shape
    mh, mw = mask.shape
    candidates: list[tuple[int, int]] = []
    for top in range(n_rows - mh + 1):
        for left in range(n_cols - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[top : top + mh, left : left + mw] & ~mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "minority placement must be unique"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the fractal overlay after inferring both the stripe background and the cell size.

    Input structure:
    - Divider rows and columns carve the board into equal square cells, but the cell size changes across samples.
    - Non-minority cells are not uniform: two recurring base tiles alternate by whole rows or by whole columns.
    - A few cells use one distinct minority tile, and that tile's full interior encodes a same-sized coarse-grid mask.

    Rule:
    - Infer the square cell step from the divider lattice, then extract every cell interior at that inferred size.
    - Identify the minority tile, convert its most frequent color into a boolean mask, and locate the unique coarse-grid
      placement consistent with the already visible minority cells.
    - Repaint the placed block so mask-True positions become the minority tile and mask-False positions revert to the
      stripe tile that the recovered row/column alternation expects there.

    Key insight:
    - The odd tile still acts as a self-similar blueprint, but it only makes sense after two hidden structures are
      recovered together: the sample-specific cell size and the A/B stripe rule of the surrounding background.
    """
    g = np.asarray(grid, dtype=int)
    step, n_cells, _, _ = _infer_cell_grid_params(g)
    inner = step - 1

    patterns, cell_pat = _extract_cell_patterns(g, step, n_cells)
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) == 3, "fusion variant expects exactly 3 cell patterns"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    _, stripe_grid = _infer_stripe_tiling(cell_pat, minority_pat)

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    assert mask.shape == (inner, inner), "minority tile must define an inner-sized mask"

    existing = np.zeros((n_cells, n_cells), dtype=bool)
    for i in range(n_cells):
        for j in range(n_cells):
            existing[i, j] = cell_pat[i][j] == minority_pat

    top, left = _find_mask_placement(existing, mask)

    out = g.copy()
    for i in range(inner):
        for j in range(inner):
            target_pat = minority_arr if mask[i, j] else patterns[stripe_grid[top + i][left + j]]
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


def _random_two_color_pattern(inner: int, color_a: int, color_b: int) -> np.ndarray:
    """Generate a structured but random-looking square tile using exactly two colors."""
    total = inner * inner
    low = max(inner, total // 3)
    high = min(total - inner, (2 * total) // 3)
    if low > high:
        low, high = 1, total - 1
    count_a = random.randint(low, high)
    coords = random.sample(range(total), count_a)
    mask = np.zeros((inner, inner), dtype=bool)
    for idx in coords:
        r, c = divmod(idx, inner)
        mask[r, c] = True
    return np.where(mask, color_a, color_b).astype(int)


def _random_stripe_patterns(inner: int, shared: int, stripe_exclusive: int) -> tuple[np.ndarray, np.ndarray]:
    """Choose two distinct square base tiles for the alternating stripe background."""
    for _ in range(80):
        pat_a = _random_two_color_pattern(inner, stripe_exclusive, shared)
        pat_b = _random_two_color_pattern(inner, stripe_exclusive, shared)
        if not np.array_equal(pat_a, pat_b):
            return pat_a, pat_b
    raise RuntimeError("failed to sample two distinct stripe patterns")


def _mask_parity_coverage(mask: np.ndarray) -> bool:
    """Require active and inactive pixels to span both row parities and both col parities."""
    active = np.argwhere(mask)
    inactive = np.argwhere(~mask)
    if active.size == 0 or inactive.size == 0:
        return False
    return (
        len({int(r) % 2 for r, _ in active}) == 2
        and len({int(c) % 2 for _, c in active}) == 2
        and len({int(r) % 2 for r, _ in inactive}) == 2
        and len({int(c) % 2 for _, c in inactive}) == 2
    )


def _random_minority_mask(inner: int, active_counts: tuple[int, ...]) -> np.ndarray:
    """Sample a square mask whose active color is dominant and visually spans the tile."""
    total = inner * inner
    coords = list(range(total))
    for _ in range(200):
        active_count = int(random.choice(active_counts))
        chosen = random.sample(coords, active_count)
        mask = np.zeros((inner, inner), dtype=bool)
        for idx in chosen:
            r, c = divmod(idx, inner)
            mask[r, c] = True
        if not mask[0, :].any():
            continue
        if not mask[-1, :].any():
            continue
        if not mask[:, 0].any():
            continue
        if not mask[:, -1].any():
            continue
        if not _mask_parity_coverage(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a valid minority mask")


def _cell_origin(i: int, j: int, step: int) -> tuple[int, int]:
    return i * step + 1, j * step + 1


def _build_base_grid(
    stripe_a: np.ndarray, stripe_b: np.ndarray, n_cells: int, orientation: str
) -> list[list[np.ndarray]]:
    """Build the full A/B assignment grid for the stripe background."""
    base: list[list[np.ndarray]] = []
    for i in range(n_cells):
        row: list[np.ndarray] = []
        for j in range(n_cells):
            stripe_idx = i if orientation == "row" else j
            row.append(stripe_a if stripe_idx % 2 == 0 else stripe_b)
        base.append(row)
    return base


def _build_grid(
    separator: int,
    base_grid: list[list[np.ndarray]],
    minority: np.ndarray,
    minority_cells: list[tuple[int, int]],
) -> np.ndarray:
    """Render the pixel grid from the stripe base plus any visible minority cells."""
    n_cells = len(base_grid)
    inner = int(base_grid[0][0].shape[0])
    step = inner + 1
    size = n_cells * step + 1
    grid = np.full((size, size), separator, dtype=int)
    for i in range(n_cells):
        for j in range(n_cells):
            paste(grid, base_grid[i][j], *_cell_origin(i, j, step))
    for i, j in minority_cells:
        paste(grid, minority, *_cell_origin(i, j, step))
    return grid


def _placement_candidates(mask: np.ndarray, given_cells: tuple[tuple[int, int], ...], n_cells: int) -> list[tuple[int, int]]:
    """Return every coarse-grid placement that can explain the shown minority cells."""
    existing = np.zeros((n_cells, n_cells), dtype=bool)
    for r, c in given_cells:
        existing[int(r), int(c)] = True

    mh, mw = mask.shape
    candidates: list[tuple[int, int]] = []
    for top in range(n_cells - mh + 1):
        for left in range(n_cells - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[top : top + mh, left : left + mw] & ~mask):
                continue
            candidates.append((top, left))
    return candidates


def _stripe_parities(cells: list[tuple[int, int]], orientation: str) -> set[int]:
    """Return which stripe parities the given coarse-grid cells occupy."""
    if orientation == "row":
        return {int(r) % 2 for r, _ in cells}
    return {int(c) % 2 for _, c in cells}


def _inactive_uses_both_stripes(mask: np.ndarray, top: int, left: int, orientation: str) -> bool:
    """Require the solved block to need both stripe tiles on mask=False positions."""
    inactive_abs = [(top + int(r), left + int(c)) for r, c in np.argwhere(~mask)]
    return len(_stripe_parities(inactive_abs, orientation)) == 2


def _choose_given_cells(
    mask: np.ndarray,
    top: int,
    left: int,
    n_cells: int,
    given_sizes: tuple[int, ...],
    orientation: str,
) -> list[tuple[int, int]] | None:
    """Choose a strict active subset that fixes the placement and still displays stripe interaction."""
    active_abs = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]
    active_idx = list(range(len(active_abs)))
    sizes = list(given_sizes)
    random.shuffle(sizes)
    for size in sizes:
        if size >= len(active_abs):
            continue
        seen: set[tuple[int, ...]] = set()
        max_trials = min(SUBSET_SAMPLE_BUDGET, len(active_abs) * 30)
        for _ in range(max_trials):
            chosen_idx = tuple(sorted(random.sample(active_idx, size)))
            if chosen_idx in seen:
                continue
            seen.add(chosen_idx)
            subset = [active_abs[idx] for idx in chosen_idx]
            if len(_stripe_parities(subset, orientation)) != 2:
                continue
            if _placement_candidates(mask, tuple(subset), n_cells) == [(top, left)]:
                return subset

        total_combos = 0
        for _ in combinations(active_idx, size):
            total_combos += 1
            if total_combos > 1200:
                break
        if total_combos <= 1200:
            valid: list[list[tuple[int, int]]] = []
            for chosen_idx in combinations(active_idx, size):
                subset = [active_abs[idx] for idx in chosen_idx]
                if len(_stripe_parities(subset, orientation)) != 2:
                    continue
                if _placement_candidates(mask, tuple(subset), n_cells) == [(top, left)]:
                    valid.append(subset)
            if valid:
                return random.choice(valid)
    return None


def _count_output_patterns(
    mask: np.ndarray, top: int, left: int, n_cells: int, orientation: str
) -> tuple[int, int, int]:
    """Count how many A, B, and M cells the solved output will contain."""
    active = {(top + int(r), left + int(c)) for r, c in np.argwhere(mask)}
    a_count = 0
    b_count = 0
    m_count = len(active)
    for i in range(n_cells):
        for j in range(n_cells):
            if (i, j) in active:
                continue
            stripe_idx = i if orientation == "row" else j
            if stripe_idx % 2 == 0:
                a_count += 1
            else:
                b_count += 1
    return a_count, b_count, m_count


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt for the fused stripe+variable-size task."""
    cfg = random.choice(SIZE_CONFIGS)
    inner = int(cfg["inner"])
    n_cells = int(cfg["n_cells"])
    active_counts = tuple(int(v) for v in cfg["active_counts"])
    given_sizes = tuple(int(v) for v in cfg["given_sizes"])

    separator, shared, stripe_exclusive, min_exclusive = random.sample(range(COLOR_POOL), 4)
    stripe_a, stripe_b = _random_stripe_patterns(inner, shared, stripe_exclusive)
    mask = _random_minority_mask(inner, active_counts)
    minority = np.where(mask, min_exclusive, shared).astype(int)
    if np.array_equal(minority, stripe_a) or np.array_equal(minority, stripe_b):
        return None

    orientation = random.choice(ORIENTATIONS)
    placements = [(top, left) for top in range(n_cells - inner + 1) for left in range(n_cells - inner + 1)]
    random.shuffle(placements)
    chosen: tuple[int, int, list[tuple[int, int]]] | None = None
    for top, left in placements:
        if not _inactive_uses_both_stripes(mask, top, left, orientation):
            continue
        a_count, b_count, m_count = _count_output_patterns(mask, top, left, n_cells, orientation)
        if not (m_count < a_count and m_count < b_count):
            continue
        subset = _choose_given_cells(mask, top, left, n_cells, given_sizes, orientation)
        if subset is None:
            continue
        chosen = (top, left, subset)
        break
    if chosen is None:
        return None

    top, left, given_cells = chosen
    active_cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]
    base_grid = _build_base_grid(stripe_a, stripe_b, n_cells, orientation)
    out = _build_grid(separator, base_grid, minority, active_cells)
    inp = _build_grid(separator, base_grid, minority, given_cells)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    step, solved_n_cells, _, _ = _infer_cell_grid_params(inp)
    patterns, cell_pat = _extract_cell_patterns(inp, step, solved_n_cells)
    distinct = {pat for row in cell_pat for pat in row}
    if len(distinct) != 3:
        return None
    minority_pat = min(
        ((pat, sum(cell == pat for row in cell_pat for cell in row)) for pat in distinct),
        key=lambda kv: kv[1],
    )[0]
    inferred_orientation, _ = _infer_stripe_tiling(cell_pat, minority_pat)
    if inferred_orientation != orientation:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
