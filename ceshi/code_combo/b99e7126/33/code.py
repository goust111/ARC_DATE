# Combo variant of b99e7126 (index 33)
# combo_spec: D3/0+D16/0
# Name: 透明度分层 + 居中放置

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 原题输入会给出少量已经落在目标 3x3 单元格块内的 minority 单元格，solver 通过它们反推出大图案的放置偏移。
#    - 本融合版本把这一步改成“单个 blueprint 单元格 + 全局居中”：
#      输入只保留 1 个 minority 单元格，它只负责提供内部 3x3 掩码，不再提供目标区域的定位信息。
#    - 之后把这张 3x3 掩码放大到单元格级时，固定居中到 7x7 单元格网格中央的 3x3 区域。
#    - 再叠加 D16 的透明层：对每个激活目标单元格，只把其 3x3 interior 的边框改成 overlay 颜色，中心像素保留底图。
#    - 小例子：若 lone minority 单元格在左上角附近，内部 mask 的 active 像素是 6 格，那么输出不会在左上角扩张；
#      它会在全局中心的 3x3 单元格块里，对对应的 6 个单元格画边框轮廓，左上角 blueprint 单元格本身保持原样。
# b) 复杂度校准
#    - 只保留一个 minority blueprint，避免旧的“由锚点求偏移”机制继续起作用，从而让 D3 真正可见。
#    - 中心目标块固定是 3x3 单元格范围，仍然由 minority 内部 3x3 mask 决定哪些单元格被描边，结构清晰且可重复观察。
#    - overlay 颜色固定为 0，输入颜色只从 1..9 采样，输出新增颜色语义稳定，不会和底图混淆。
# c) 原 main() 旧假设审查
#    - 原先的 candidate placement 搜索依赖“输入 minority 锚点已经落在目标块里”；融合后这个前提被故意打破，必须删掉该搜索。
#    - “minority 内部主色对应 active mask” 这个前提仍然成立，因为 generator 继续让 minority-exclusive 颜色占 6 或 7 个像素。
#    - “输出直接把目标单元格改写成 minority pattern” 不再成立；融合后只在激活单元格里画边框层。
# d) 放置合法性
#    - 7x7 网格配 3x3 mask 时，居中放置唯一对应 top=left=2；不会越界，也不接触分隔线。
#    - generator 强制 lone blueprint 单元格落在中心 3x3 目标块之外，避免与居中叠加区域重合，视觉上更容易区分“源”和“目标”。
#    - overlay 仅写入单元格 interior 的边框像素，不会跨格重叠。
# e) 视觉可观察性
#    - 多个样例里，odd cell 会出现在不同偏离中心的位置，但输出轮廓总是出现在网格中央。
#    - 这让解题者能从 2-3 组例子归纳出：odd cell 只提供图案内容，定位规则来自全局居中，而不是 odd cell 的坐标。
# f) 可视化整洁度
#    - 中心 overlay 只使用一种新颜色，且只画边框；内部底图还在，因此不会把 cell pattern 信息完全抹掉。
#    - blueprint 单元格与中心目标块被空间上分离，减少多层叠加导致的杂乱。
# g) 颜色变化可推断性
#    - 输出的新颜色只出现在中心激活单元格的边界上，条件由几何位置和 mask 结构共同决定，而不是颜色 id 的隐式排序。
# h) 新的隐式假设
#    - solver 假设输入恰有一个最少见的单元格图案，generator 用“48 个 majority + 1 个 minority”保证这点。
#    - solver 假设内层尺寸至少为 3，才能形成“边框着色 + 中心透明”的效果；本题固定 inner=3。


OVERLAY_COLOR = 0

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = CELL_INNER + 1
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
MASK_SIZE = CELL_INNER
ACTIVE_PIXEL_COUNTS = (6, 7)
INPUT_COLORS = list(range(1, 10))
MAX_RETRIES = 300

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    """Infer the divider step from full-width/height separator lines."""
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need at least 2 row and 2 col dividers"
    row_divs.sort()
    col_divs.sort()
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1)]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1)]
    step = reduce(gcd, drs + dcs)
    assert step > 1, "divider step must be > 1"
    assert row_divs[0] == 0 and col_divs[0] == 0, "first divider must be at index 0"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "last divider must be at boundary"
    assert (h - 1) % step == 0 and (w - 1) % step == 0, "grid dimensions must be (k*step + 1)"
    return int(step), row_divs, col_divs


def _extract_cell_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Read every cell interior and cache the distinct patterns."""
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
            pat = tuple(int(x) for x in pat_arr.ravel())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)
    return patterns, cell_pat


def _infer_minority_pattern(g: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the minority interior pattern and the inferred cell step."""
    step, _, _ = _infer_cell_grid_params(g)
    patterns, cell_pat = _extract_cell_patterns(g, step)
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns"
    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    return patterns[minority_pat], step


def _minority_mask(minority_arr: np.ndarray) -> np.ndarray:
    """Foreground mask from the minority cell's dominant color."""
    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    return minority_arr == active_color


def _centered_offset(n: int, span: int) -> int:
    """Center a span of cells within an n-cell axis."""
    assert n >= span, "span must fit inside the cell grid"
    return int((n - span) // 2)


def _cell_frame_mask(inner: int) -> np.ndarray:
    """Boolean mask selecting only the border of a cell interior."""
    frame = np.ones((inner, inner), dtype=bool)
    if inner > 2:
        frame[1:-1, 1:-1] = False
    return frame


def _apply_centered_overlay(g: np.ndarray, mask: np.ndarray, step: int) -> np.ndarray:
    """Draw the active mask as centered cell-border overlays."""
    n = (g.shape[0] - 1) // step
    inner = step - 1
    top = _centered_offset(n, mask.shape[0])
    left = _centered_offset(n, mask.shape[1])
    frame = _cell_frame_mask(inner)

    out = g.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i, j]:
                continue
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            cell = out[r0 : r0 + inner, c0 : c0 + inner]
            cell[frame] = OVERLAY_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the odd cell as a stencil, enlarge it to the cell lattice, and render the result as a centered translucent outline.

    Input structure:
    - A 29x29 grid is split into a 7x7 array of 3x3 cell interiors by single-color divider lines.
    - Almost every cell repeats one majority interior pattern.
    - Exactly one off-center cell uses a different minority interior pattern; that lone cell is the blueprint.

    Rule:
    - Extract the minority cell's 3x3 dominant-color mask.
    - Ignore the blueprint cell's location when placing the enlarged mask; instead center the 3x3 cell-sized mask on the whole 7x7 cell grid.
    - For each active centered target cell, keep the underlying interior pattern but recolor only its border pixels with the overlay color.

    Key insight:
    The odd cell still encodes which positions are active, but no longer tells you where the enlarged figure goes.
    Shape comes from the minority pattern; position comes from global centering; the final drawing is shown as a separate outline layer rather than a hard overwrite.
    """
    g = np.asarray(grid, dtype=int)
    minority_arr, step = _infer_minority_pattern(g)
    mask = _minority_mask(minority_arr)
    return _apply_centered_overlay(g, mask, step)


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate valid LR-symmetric 3x3 masks keyed by active count."""
    weights = [len(group) for group in _LR_GROUPS]
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
                if mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform LR-symmetric 3x3 pattern using two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(color == choices[0] for color in choices):
        flip = color_b if choices[0] == color_a else color_a
        choices[random.randrange(len(choices))] = flip
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask() -> np.ndarray:
    """Pick a random valid minority mask."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Pixel origin of cell (i, j) interior."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, minority: np.ndarray, source_cell: tuple[int, int]) -> np.ndarray:
    """Construct the input grid with one minority blueprint cell."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    paste(grid, minority, *_cell_origin(*source_cell))
    return grid


def _center_block_cells() -> set[tuple[int, int]]:
    """All cells inside the centered 3x3 placement window."""
    top = _centered_offset(CELL_GRID_N, MASK_SIZE)
    left = _centered_offset(CELL_GRID_N, MASK_SIZE)
    return {(top + i, left + j) for i in range(MASK_SIZE) for j in range(MASK_SIZE)}


_CENTER_BLOCK_CELLS = _center_block_cells()
_SOURCE_CANDIDATES = [
    (i, j)
    for i in range(CELL_GRID_N)
    for j in range(CELL_GRID_N)
    if (i, j) not in _CENTER_BLOCK_CELLS
]


def _expected_output(inp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the centered overlay rule used by the variant."""
    return _apply_centered_overlay(inp, mask, CELL_STEP)


def _gen_attempt():
    """Build one input and verify that the solver reproduces the intended output."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(INPUT_COLORS, 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    source_cell = random.choice(_SOURCE_CANDIDATES)
    inp = _build_grid(separator, majority, minority, source_cell)
    expected = _expected_output(inp, mask)

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
