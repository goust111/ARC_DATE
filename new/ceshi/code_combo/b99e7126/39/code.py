# Combo variant of b99e7126 (index 39)
# combo_spec: D1/0+D6/1+D12/1
# Name: 频率排序 + 行间镜像 + 条带式铺贴

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    原题里所有普通格子都相同，只找唯一少数格；这里改成“整行条带 + 行间镜像 + 频率排序”。
#    具体地，背景家族 B 在偶数行使用一个 3x3 纹理，在奇数行使用它的水平镜像，因此整板底纹
#    呈现 B / mirror(B) / B / mirror(B) ... 的水平条带。与此同时，还会散落两个干扰家族 D1/D2
#    和一个源家族 S；它们也都遵守同样的行镜像规则：若某家族的偶数行版本为 P，则奇数行版本
#    固定为 mirror(P)。输入里真正要被放大的不是“唯一异常 exact pattern”，而是镜像归并后的
#    最低频家族 S。输出时，S 的偶数行版本内部 3x3 像素掩码会在 7x7 单元格网格上放大；掩码
#    为 True 的位置写入 S 的对应行版本，False 的位置恢复为背景家族 B 的对应行版本。
#    例如，若源家族在输入中以 2 个偶数行格 + 1 个奇数行格出现，则 exact pattern 计数会看到
#    两种不同纹理，但镜像归并后它们其实是同一家族，总频次为 3。
# b) Complexity calibration:
#    只保留 4 个镜像家族：背景、源、两个干扰。这样频率排序有足够层次（source < d1 < d2 < bg），
#    但自由度仍可控。背景条带方向固定为“按行交替”，避免再引入横/纵双分支；镜像只用水平翻转，
#    让推理链集中在“先归并家族，再定位分形块，再按行 parity 还原纹理”。
# c) Broken assumptions from the original solver:
#    - `minority_pat = min(counts.items(), ...)` 不再成立，因为同一家族会被奇偶行拆成两个 exact pattern。
#    - `majority_pat = max(counts.items(), ...)` 也不再安全；背景的偶/奇行版本 exact pattern 不同，
#      必须先做镜像归并才能知道哪个家族是真正背景。
#    - `target_pat = minority if mask else majority` 会抹掉行间镜像关系；这里 False 位必须按目标行的
#      parity 恢复为背景家族的偶/奇版本，True 位也要按目标行 parity 写入源家族版本。
# d) Placement legality:
#    生成器仍用 output-first。先铺满背景条带，再放完整源块，最后把两个干扰家族散布到 3x3 完成块
#    外部。所有坐标都在固定 7x7 cell grid 内采样，互不重叠、不越界。输入只保留源块 active 位中的
#    3-4 个已知格，其余 active 位回退成背景。
# e) Visual observability:
#    人眼能直接看到每一整行都在 B 与 mirror(B) 之间交替；源家族和干扰家族一旦跨越奇偶行，就会以
#    左右翻转的两种 exact pattern 成对出现。因此“把镜像当同一类再计数”是画面上可见的，不依赖颜色 id。
#    同时，源家族内部掩码故意选成非水平对称，保证奇偶行源格的镜像关系肉眼明显。
# f) Visual cleanliness:
#    颜色总数控制在 6 个：divider、shared、背景专属、源专属、两个干扰专属。每个家族只用 shared +
#    一个专属色，画面虽然有频率排序层次，但纹理和颜色来源仍然清楚，不会和分隔线混在一起。
# g) Change inferability:
#    本变体没有任何基于颜色数值大小的排序。频率比较完全基于“镜像归并后的家族出现次数”，而镜像关系、
#    行条带、以及输出里 False 位恢复背景条带的行为，都能从少量样例中由可见结构推断出来。
# h) Anti-degeneration guard:
#    生成器强制源家族、两个干扰家族、以及 3x3 完成块的 active/inactive 行集合都跨越两种行 parity。
#    这样若 solver 不做镜像归并，往往只能看到被拆开的 half-family；若 solver 忽略条带并把 False 位
#    统一刷成一个 exact pattern，也会与正确输出不同。
# i) Solver assumptions and generator guarantees:
#    - 镜像归并后恰有 4 个家族，且频次严格满足 source < d1 < d2 < bg。
#    - 每个家族在奇数行的 exact pattern 都是偶数行版本的水平镜像。
#    - 源家族的 active 色是其偶数行版本中的唯一多数色，且其 3x3 掩码唯一确定放置位置。
#    这些约束都由生成器显式构造，并通过 `main(inp) == out` 的回归检查兜底。


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
SOURCE_GIVEN_SIZES = (3, 4)
COLOR_POOL = 10
MAX_RETRIES = 600


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


def _hmirror(arr: np.ndarray) -> np.ndarray:
    """Horizontal mirror of a 2D pattern."""
    return np.fliplr(arr)


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    """Flatten a cell pattern into a hashable exact-pattern key."""
    return tuple(int(v) for v in arr.ravel())


def _family_key(arr: np.ndarray) -> tuple[int, ...]:
    """Canonicalize a pattern modulo horizontal mirroring."""
    key = _pattern_key(arr)
    mirrored = _pattern_key(_hmirror(arr))
    return key if key <= mirrored else mirrored


def _extract_cell_arrays(g: np.ndarray, step: int) -> list[list[np.ndarray]]:
    """Extract each cell interior as a 3x3 array."""
    n = (g.shape[0] - 1) // step
    inner = step - 1
    cell_arrs: list[list[np.ndarray]] = []
    for i in range(n):
        row: list[np.ndarray] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            row.append(g[r0 : r0 + inner, c0 : c0 + inner].copy())
        cell_arrs.append(row)
    return cell_arrs


def _count_families(cell_arrs: list[list[np.ndarray]]) -> dict[tuple[int, ...], int]:
    """Count motif families after merging horizontal mirrors."""
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_arrs:
        for arr in row:
            fam = _family_key(arr)
            counts[fam] = counts.get(fam, 0) + 1
    return counts


def _family_mask(cell_arrs: list[list[np.ndarray]], family: tuple[int, ...]) -> np.ndarray:
    """Boolean mask of where a canonical family appears in the cell grid."""
    n = len(cell_arrs)
    out = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            out[i, j] = _family_key(cell_arrs[i][j]) == family
    return out


def _family_patterns_by_parity(
    cell_arrs: list[list[np.ndarray]], family: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover a family's even-row and odd-row exact patterns.

    Odd rows are guaranteed to be the horizontal mirror of even rows.
    If one parity is absent from the observation set, infer it by mirroring the other.
    """
    parity_seen: dict[int, np.ndarray] = {}
    for i, row in enumerate(cell_arrs):
        for arr in row:
            if _family_key(arr) != family:
                continue
            parity = i % 2
            if parity in parity_seen:
                assert np.array_equal(parity_seen[parity], arr), "family must be row-consistent within parity"
            else:
                parity_seen[parity] = arr.copy()

    assert parity_seen, "family must appear at least once"
    if 0 not in parity_seen:
        parity_seen[0] = _hmirror(parity_seen[1])
    if 1 not in parity_seen:
        parity_seen[1] = _hmirror(parity_seen[0])
    assert np.array_equal(parity_seen[1], _hmirror(parity_seen[0])), "odd rows must mirror even rows"
    return parity_seen[0], parity_seen[1]


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a row-striped fractal by ranking mirror-equivalent motif families.

    Input structure:
    - The board is partitioned into a 7x7 grid of 3x3 cell interiors separated by divider lines.
    - Most cells belong to one dominant background family that forms horizontal stripes:
      even rows use one pattern and odd rows use its horizontal mirror.
    - Three rarer motif families also appear; the true source family is not an exact-pattern oddity,
      because its even-row and odd-row appearances are mirror images of one another.

    Rule:
    - Group cell patterns into families by treating a pattern and its horizontal mirror as the same type.
    - Choose the uniquely least frequent family as the source blueprint, recover its even-row version,
      and read its unique majority color as a boolean 3x3 mask.
    - Find the unique placement where all shown source-family cells fit that mask, then fill mask-True
      positions with the source family's row-parity version and mask-False positions with the background
      family's row-parity version.

    Key insight:
    - The counting step must happen at the mirror-family level, not at the raw exact-pattern level:
      odd rows visually flip the same family.
    - After that normalization, the task reverts to a fractal completion, but the completed block must
      preserve the row-striped mirror background instead of collapsing to one uniform majority tile.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    cell_arrs = _extract_cell_arrays(g, step)
    family_counts = _count_families(cell_arrs)
    assert len(family_counts) >= 4, "need at least 4 mirror-families"

    ordered = sorted(family_counts.items(), key=lambda kv: kv[1])
    assert ordered[0][1] < ordered[1][1], "source family frequency must be uniquely minimal"
    assert ordered[-1][1] > ordered[-2][1], "background family frequency must be uniquely maximal"
    source_family = ordered[0][0]
    background_family = ordered[-1][0]

    source_even, source_odd = _family_patterns_by_parity(cell_arrs, source_family)
    background_even, background_odd = _family_patterns_by_parity(cell_arrs, background_family)

    vals, cnts = np.unique(source_even, return_counts=True)
    order = np.argsort(cnts)[::-1]
    assert len(order) >= 2 and cnts[order[0]] > cnts[order[1]], "source pattern needs a unique active color"
    active_color = int(vals[order[0]])
    mask = source_even == active_color
    mh, mw = mask.shape

    existing = _family_mask(cell_arrs, source_family)

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

    assert len(candidates) == 1, "source family must admit a unique mask placement"
    top, left = candidates[0]

    out = g.copy()
    for i in range(mh):
        row_source = source_even if (top + i) % 2 == 0 else source_odd
        row_background = background_even if (top + i) % 2 == 0 else background_odd
        for j in range(mw):
            target_pat = row_source if mask[i, j] else row_background
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


def _precompute_source_masks() -> dict[int, list[np.ndarray]]:
    """All valid non-LR-symmetric source masks keyed by active-pixel count."""
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        masks: list[np.ndarray] = []
        for combo in combinations(range(MASK_SIZE * MASK_SIZE), active_count):
            mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
            for idx in combo:
                r, c = divmod(idx, MASK_SIZE)
                mask[r, c] = True
            if np.array_equal(mask, _hmirror(mask)):
                continue
            if not mask[0, :].any() or not mask[-1, :].any():
                continue
            if not mask[:, 0].any() or not mask[:, -1].any():
                continue
            masks.append(mask)
        by_count[active_count] = masks
    return by_count


def _precompute_pattern_masks() -> list[np.ndarray]:
    """Reusable non-mirror 3x3 boolean masks for background and distractors."""
    masks: list[np.ndarray] = []
    for on_count in range(3, 7):
        for combo in combinations(range(MASK_SIZE * MASK_SIZE), on_count):
            mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
            for idx in combo:
                r, c = divmod(idx, MASK_SIZE)
                mask[r, c] = True
            if np.array_equal(mask, _hmirror(mask)):
                continue
            masks.append(mask)
    return masks


_SOURCE_MASKS_BY_COUNT = _precompute_source_masks()
_PATTERN_MASKS = _precompute_pattern_masks()


def _random_source_mask() -> np.ndarray:
    """Pick one valid source mask with 6 or 7 active pixels."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_SOURCE_MASKS_BY_COUNT[active_count]).copy()


def _random_nonmirror_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a 3x3 two-color pattern that visibly changes under horizontal mirroring."""
    mask = random.choice(_PATTERN_MASKS)
    return np.where(mask, color_b, color_a).astype(int)


def _row_variant(even_pattern: np.ndarray, row_idx: int) -> np.ndarray:
    """Exact pattern for the given row parity."""
    return even_pattern if row_idx % 2 == 0 else _hmirror(even_pattern)


def _row_parities(rel_cells: list[tuple[int, int]], top: int) -> set[int]:
    """Which absolute row parities are occupied by relative cell positions."""
    return {(top + r) % 2 for r, _ in rel_cells}


def _placement_candidates(mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    """Find all placements consistent with the shown source subset."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True

    candidates: list[tuple[int, int]] = []
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
    """Choose 3-4 visible source cells that uniquely determine the placement and span both row parities."""
    active = [tuple(int(v) for v in coord) for coord in np.argwhere(mask)]
    sizes = list(SOURCE_GIVEN_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(active, size)
            if _placement_candidates(mask, subset, top, left) == [(top, left)]
            and len(_row_parities(list(subset), top)) == 2
        ]
        if valid:
            return random.choice(valid)
    return None


def _variant_visible(mask: np.ndarray, top: int) -> bool:
    """Require both source cells and recovered background cells to involve both row parities."""
    active = [tuple(int(v) for v in coord) for coord in np.argwhere(mask)]
    inactive = [tuple(int(v) for v in coord) for coord in np.argwhere(~mask)]
    return len(_row_parities(active, top)) == 2 and len(_row_parities(inactive, top)) == 2


def _positions_outside_block(top: int, left: int) -> list[tuple[int, int]]:
    """All cell coordinates outside the 3x3 completion block."""
    return [
        (i, j)
        for i in range(CELL_GRID_N)
        for j in range(CELL_GRID_N)
        if not (top <= i < top + MASK_SIZE and left <= j < left + MASK_SIZE)
    ]


def _sample_cells_with_both_parities(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    """Sample distinct cells whose rows cover both parities."""
    if count < 2:
        return None
    for _ in range(100):
        chosen = random.sample(cells, count)
        if len({r % 2 for r, _ in chosen}) == 2:
            return chosen
    return None


def _build_grid(
    separator: int,
    background_even: np.ndarray,
    family_even_patterns: dict[str, np.ndarray],
    family_cells: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    """Construct the full grid from the striped background and family overrides."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        bg_pat = _row_variant(background_even, i)
        for j in range(CELL_GRID_N):
            paste(grid, bg_pat, *_cell_origin(i, j))

    for name, even_pat in family_even_patterns.items():
        for i, j in family_cells[name]:
            paste(grid, _row_variant(even_pat, i), *_cell_origin(i, j))
    return grid


def _gen_attempt() -> np.ndarray | None:
    """Build one valid fused input grid."""
    separator, shared, bg_color, src_color, d1_color, d2_color = random.sample(range(COLOR_POOL), 6)

    background_even = _random_nonmirror_pattern(shared, bg_color)
    mask = _random_source_mask()
    source_even = np.where(mask, src_color, shared).astype(int)
    distractor1_even = _random_nonmirror_pattern(shared, d1_color)
    distractor2_even = _random_nonmirror_pattern(shared, d2_color)

    family_patterns = [background_even, source_even, distractor1_even, distractor2_even]
    family_keys = {_family_key(p) for p in family_patterns}
    if len(family_keys) != 4:
        return None

    placements = [(top, left) for top in range(PLACEMENT_RANGE) for left in range(PLACEMENT_RANGE)]
    random.shuffle(placements)
    chosen: tuple[int, int, tuple[tuple[int, int], ...]] | None = None
    for top, left in placements:
        if not _variant_visible(mask, top):
            continue
        subset = _choose_given_subset(mask, top, left)
        if subset is not None:
            chosen = (top, left, subset)
            break
    if chosen is None:
        return None

    top, left, subset = chosen
    active_cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]
    given_cells = [(top + r, left + c) for r, c in subset]

    source_count = len(given_cells)
    distractor1_count = source_count + random.choice((2, 3))
    distractor2_count = distractor1_count + random.choice((2, 3))

    free_cells = _positions_outside_block(top, left)
    d1_cells = _sample_cells_with_both_parities(free_cells, distractor1_count)
    if d1_cells is None:
        return None
    remaining = [cell for cell in free_cells if cell not in set(d1_cells)]
    d2_cells = _sample_cells_with_both_parities(remaining, distractor2_count)
    if d2_cells is None:
        return None

    input_family_cells = {
        "source": given_cells,
        "d1": d1_cells,
        "d2": d2_cells,
    }
    output_family_cells = {
        "source": active_cells,
        "d1": d1_cells,
        "d2": d2_cells,
    }
    family_even_patterns = {
        "source": source_even,
        "d1": distractor1_even,
        "d2": distractor2_even,
    }

    inp = _build_grid(separator, background_even, family_even_patterns, input_family_cells)
    out = _build_grid(separator, background_even, family_even_patterns, output_family_cells)

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
