# Combo variant of b99e7126 (index 50)
# combo_spec: D5/1+D16/0
# Name: 颜色混合叠加 + 透明度分层

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) 具体化与行为追踪
#    - 原题：minority 单元格内部的 3x3 图案既是蓝图，也是输出时要整块复制到
#      放大区域每个 active 位置上的目标单元格内容。
#    - D5/1 单变体：active 位置不再复制 raw minority，而是换成一个“混色后”的
#      第三类单元格；这让输入必须额外提供 mixed preview 作为颜色规则样例。
#    - D16/0 单变体：叠加不再覆盖整个前景，只保留“边界层”，内部维持底图。
#    - 本融合版把两者合并为一个更受控的可视规则：
#      1. minority 单元格仍然提供 3x3 的 active mask。
#      2. mask 放大到 7x7 单元格网格后，只把 active 区域的“外圈单元格”视作
#         叠加边界；完全被四邻 active 包围的内部 active 单元格不着色。
#      3. 边界单元格也不是 raw minority，而是一个边框为 mix_color 的 mixed
#         preview；其中心像素保留 majority。于是视觉上得到彩色边框。
#    - 小例子：
#      若 local mask 是
#          0 1 0
#          1 1 1
#          1 1 1
#      则中心 (1,1) 是内部 active。原题会把这 7 个 active 单元格全部改成
#      minority；本融合版只把外圈 6 个单元格改成 mixed preview，而中心单元格
#      在输出中恢复为 majority。输入里如果中心位置给的是 minority clue，
#      输出会把它“擦回去”，这正是透明分层带来的差异。
#
# b) 复杂度校准
#    - 仍使用原题固定的 7x7 单元格格局、3x3 单元格内部、3x3 fractal mask。
#    - 输入只保留三种单元格模式：majority、minority clue、boundary-mixed
#      preview。避免为不同朝向的轮廓边单独造模式，否则模式种类会爆炸。
#    - mixed preview 的像素级样式固定为“3x3 边框改成 mix_color，中心保留
#      majority”，让人类可以跨样例稳定识别“这是混色后的边界层”。
#    - generator 只挑选至少有 1 个内部 active 的 mask，保证输出中一定存在
#      “active 但不混色”的位置，避免退化成 D5 单变体那种所有 active 都混色。
#
# c) 原 main() 隐式假设审查
#    - 原代码假设输入里只有 majority/minority 两类单元格；这里不再成立，必须改为
#      识别三类模式，并要求频数满足 minority < mixed < majority。
#    - 原代码把所有 minority 位置视为 active 证据；这里 mixed preview 也同样是
#      active 证据，否则只看到边界预览时无法唯一定位。
#    - 原代码在输出阶段对每个 active 位置直接写 minority_arr；这里改成：
#      boundary_active -> mixed_arr，interior_active / inactive -> majority_arr。
#    - 原代码不需要检查 mixed 与 majority 的关系；这里必须断言 mixed 只改 frame，
#      中心保持 majority，否则“透明分层”语义不成立。
#
# d) 放置合法性
#    - mask 仍限制在 7x7 中的 0..4 偏移，3x3 block 永不越界。
#    - 所有写入都以单元格 interior 为单位，不碰 divider 线。
#    - 输入中的 mixed previews 只取自 boundary active；minority clues 取自 active。
#      两者都是 active 子集，所以不会出现与题意冲突的“假边界”。
#
# e) 视觉可观察性
#    - 训练样例中，人类会看到：
#      majority 底图、少数 raw minority clue、以及少数彩色边框 mixed preview。
#    - 输出时，mixed preview 会扩展成整个放大块的外圈，而某些 clue 单元格会恢复
#      成底图，说明“蓝图决定区域，但真正叠加只显示边界层”。
#    - 颜色变化不依赖颜色编号，只依赖结构角色：只有 boundary active 会变成
#      mixed preview，内部 active 不会。
#
# f) 可视化整洁度
#    - 叠加层只占 3x3 单元格的 8 个边框像素，中心保留底图，因此不会把 majority
#      的纹理完全抹掉。
#    - 由于只高亮单元格级外圈，不给内部 active 着色，输出会形成清晰的一圈彩边，
#      而不是整块杂色。
#
# g) 变化的可推断性
#    - 每个实例只引入一个 mix_color，且它在输入的 preview 与输出的 completed
#      boundary 中重复出现，不需要解题者猜大型映射表。
#    - 生成器固定要求至少一个内部 active clue，从而保证“只混边界、不混内部”在
#      样例里有直接证据，而不是隐藏在不可见状态里。
#
# h) 新的隐式假设与 generator 保证
#    - solver 假设 mixed preview 与 majority 的中心像素相同，且 frame 全部改色；
#      generator 明确按此构造 mixed。
#    - solver 假设 minority 的 dominant color 定义 active mask；generator 保证
#      minority active 像素数为 7，且该颜色严格占多数。
#    - solver 允许任意 mask 的 boundary / interior 拆分；generator 进一步只采样
#      含内部 active 的 mask，并要求 evidence 子集唯一定位。
# ---------------------------------------------------------------------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_COUNT_PAIRS = ((1, 2), (1, 3), (2, 3))
COLOR_POOL = 10
MAX_RETRIES = 400

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate all valid LR-symmetric 3x3 masks, keyed by active count."""
    weights = [len(g) for g in _LR_GROUPS]
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
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


def _extract_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Collect every cell interior pattern and its array form."""
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


def _pattern_counts(cell_pat: list[list[tuple[int, ...]]]) -> dict[tuple[int, ...], int]:
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    return counts


def _cell_frame_mask(inner: int) -> np.ndarray:
    """Border pixels of a cell interior; the center remains transparent."""
    frame = np.ones((inner, inner), dtype=bool)
    if inner > 2:
        frame[1:-1, 1:-1] = False
    return frame


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Active pixels that touch the outside in 4-neighborhood."""
    boundary = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not mask[rr, cc]:
                    boundary[r, c] = True
                    break
    return boundary


def _layout_from_input(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Infer majority/minority/mixed patterns, the active mask, and its placement."""
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns, cell_pat = _extract_patterns(g, step)
    counts = _pattern_counts(cell_pat)
    assert len(counts) == 3, "variant expects exactly three cell patterns"

    ranked = sorted(counts.items(), key=lambda kv: kv[1])
    minority_pat = ranked[0][0]
    mixed_pat = ranked[1][0]
    majority_pat = ranked[2][0]
    assert counts[minority_pat] < counts[mixed_pat] < counts[majority_pat], "pattern frequencies must be ordered"

    minority_arr = patterns[minority_pat]
    mixed_arr = patterns[mixed_pat]
    majority_arr = patterns[majority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    mh, mw = mask.shape
    assert mh == inner and mw == inner, "minority mask must match cell interior"
    assert mask.sum() >= 6, "minority active color should dominate the 3x3 blueprint"

    frame = _cell_frame_mask(inner)
    assert np.array_equal(mixed_arr[~frame], majority_arr[~frame]), "mixed preview must keep the majority center"
    assert np.all(mixed_arr[frame] != majority_arr[frame]), "mixed preview must recolor the whole frame"
    assert np.all(mixed_arr[frame] != minority_arr[frame]), "mixed frame color must differ from the raw minority clue"

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] in (minority_pat, mixed_pat)

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
    assert len(candidates) == 1, "must have a unique mask placement aligning all active evidence"
    top, left = candidates[0]

    return majority_arr, minority_arr, mixed_arr, mask, step, top, left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a fractal placement from the odd cell, then render only its mixed-color outer ring.

    Input structure:
    - A 7x7 lattice of 3x3 cell interiors is separated by one-color divider lines.
    - Most cells show one majority pattern.
    - A few cells show the raw minority blueprint, and a few others already show the
      mixed border style that marks the completed overlay's boundary.

    Rule:
    - Read the minority cell's dominant color as a 3x3 active mask.
    - Place that mask uniquely on the 7x7 cell lattice using both raw minority clues
      and mixed preview cells as evidence for active positions.
    - Among the placed active positions, recolor only the boundary cells with the
      mixed preview pattern; leave inactive cells and fully enclosed active cells as majority.

    Key insight:
    - The odd cell still encodes the large-scale fractal footprint, but the finished
      image is a layered outline: the shape's interior falls back to the base pattern,
      while only its perimeter carries the mixed-color border.
    """
    g = np.asarray(grid, dtype=int)
    majority_arr, _, mixed_arr, mask, step, top, left = _layout_from_input(g)
    boundary = _boundary_mask(mask)

    out = g.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            target_pat = mixed_arr if boundary[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + CELL_INNER, c0 : c0 + CELL_INNER] = target_pat
    return out


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a random LR-symmetric 3x3 pattern using two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _eligible_masks() -> list[np.ndarray]:
    """Masks whose enlarged region has a non-empty interior, avoiding D5-style degeneration."""
    masks: list[np.ndarray] = []
    for variants in _MASKS_BY_COUNT.values():
        for mask in variants:
            boundary = _boundary_mask(mask)
            if np.any(mask & ~boundary):
                masks.append(mask)
    return masks


_COMBO_MASKS = _eligible_masks()


def _random_minority_mask() -> np.ndarray:
    return random.choice(_COMBO_MASKS)


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_mixed_preview(majority: np.ndarray, mix_color: int) -> np.ndarray:
    """Create the mixed border style used for boundary cells."""
    mixed = majority.copy()
    mixed[_cell_frame_mask(CELL_INNER)] = mix_color
    return mixed


def _build_grid(
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    mixed: np.ndarray,
    minority_cells: list[tuple[int, int]],
    mixed_cells: list[tuple[int, int]],
) -> np.ndarray:
    """Construct a full grid from majority cells plus minority/mixed exceptions."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in minority_cells:
        paste(grid, minority, *_cell_origin(i, j))
    for i, j in mixed_cells:
        paste(grid, mixed, *_cell_origin(i, j))
    return grid


def _placement_candidates(mask: np.ndarray, evidence_subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    """Find all offsets where the observed active-evidence subset fits the mask."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in evidence_subset:
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


def _choose_evidence_partition(mask: np.ndarray, top: int, left: int):
    """
    Choose visible active positions for the input.

    - At least one interior active is shown as a raw minority clue, so the final
      output visibly differs from an "all active cells become mixed" rule.
    - Mixed previews are sampled only from boundary active positions.
    """
    boundary = _boundary_mask(mask)
    active = [tuple(coord) for coord in np.argwhere(mask)]
    interior = [pos for pos in active if not boundary[pos]]
    boundary_cells = [pos for pos in active if boundary[pos]]
    if not interior:
        return None

    anchor = random.choice(interior)
    pair_options = list(GIVEN_COUNT_PAIRS)
    random.shuffle(pair_options)
    for clue_count, preview_count in pair_options:
        if preview_count >= len(boundary_cells):
            continue
        extra_clues_needed = clue_count - 1
        if extra_clues_needed < 0:
            continue

        preview_options = list(combinations(boundary_cells, preview_count))
        random.shuffle(preview_options)
        for mixed_subset in preview_options:
            remaining = [pos for pos in active if pos != anchor and pos not in mixed_subset]
            if extra_clues_needed > len(remaining):
                continue
            clue_options = list(combinations(remaining, extra_clues_needed))
            random.shuffle(clue_options)
            for extra_clues in clue_options:
                minority_subset = (anchor,) + extra_clues
                evidence = tuple(minority_subset) + tuple(mixed_subset)
                if _placement_candidates(mask, evidence, top, left) == [(top, left)]:
                    return tuple(minority_subset), tuple(mixed_subset)
    return None


def _expected_output(
    separator: int,
    majority: np.ndarray,
    mixed: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
) -> np.ndarray:
    """Construct the intended combo output from the latent placement."""
    boundary = _boundary_mask(mask)
    mixed_cells = [(top + r, left + c) for r, c in np.argwhere(boundary)]
    return _build_grid(separator, majority, majority, mixed, [], mixed_cells)


def _gen_attempt():
    """
    Single generation attempt.

    Output-first construction:
    - Output shows mixed-color border cells on the active region's outer ring only.
    - Input reveals a raw interior clue plus a few boundary previews, so both the
      mask placement and the fused rendering rule are visible.
    """
    separator, shared, maj_exclusive, min_exclusive, mix_color = random.sample(range(COLOR_POOL), 5)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)
    mixed = _build_mixed_preview(majority, mix_color)

    if np.array_equal(majority, minority) or np.array_equal(majority, mixed) or np.array_equal(minority, mixed):
        return None

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    expected = _expected_output(separator, majority, mixed, mask, top, left)

    partition = _choose_evidence_partition(mask, top, left)
    if partition is None:
        return None
    minority_subset, mixed_subset = partition
    minority_cells = [(top + r, left + c) for r, c in minority_subset]
    mixed_cells = [(top + r, left + c) for r, c in mixed_subset]
    inp = _build_grid(separator, majority, minority, mixed, minority_cells, mixed_cells)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
