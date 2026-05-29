# Variant of b99e7126
# Dimension: D16 — 叠加后的视觉效果处理 (transform)
# Change: 简单覆盖——放大图案的前景区域直接覆写原网格对应位置... → 透明度分层：放大图案作为半透明层叠加

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# Design analysis for the D16 variant
# -----------------------------------
# a) 具体化与行为追踪
#    - 输入阶段保持原题机制不变：29x29 网格由 7x7 个 3x3 单元格组成，分隔线单色。
#    - 绝大多数单元格是同一个 majority 图案，少数已给出的锚点单元格是 minority 图案。
#    - minority 单元格内部的 3x3 像素仍然充当“蓝图掩码”；solver 仍然要从它恢复 3x3 的单元格级放置。
#    - D16 的变化只发生在 transform 的最后一步：
#      原题会把 mask=True 的目标单元格整块改写成 minority 图案；
#      现在改成在这些目标单元格内部加一层描边框，边框颜色是专用 overlay 颜色，中心像素保留底图。
#    - 小例子追踪：
#      若某个目标位置原本是 majority 单元格 `abc/def/ghi`，原题输出会整块变成 minority；
#      新题输出会变成 `ooo/ofo/ooo`（o=overlay 颜色），也就是仅边界改色，中心 `f` 保留。
# b) 复杂度校准
#    - 仍然只有一个 3x3 mask、一个唯一放置、3-4 个已知锚点，主推理复杂度与原题一致。
#    - 为了让“半透明层”肉眼可见，overlay 颜色固定且不出现在输入中，避免与底图颜色混淆。
#    - 每个激活单元格都被描边，而不是只画整体外轮廓；这样 6-7 个激活位置都会显式可见，不会退化成少量外边缘标记。
# c) 原 main() 中依赖旧属性的代码审查
#    - 网格解析、majority/minority 统计、从 minority 最频繁颜色提取 mask，这些都依赖输入仍是两类底图单元格；
#      本变体保持输入不变，因此这些前提继续成立。
#    - 原先 `target_pat = minority_arr if mask[i, j] else majority` 的整块覆写依赖“前景应完全取代底图”；
#      D16 取消了这个前提，所以这里必须改成“只修改目标单元格的边框像素”。
#    - 由于输出引入新颜色，solver 不能再从输出反推输入结构；但 main() 只处理输入，所以无影响。
# d) 放置合法性
#    - generator 继续限制 mask 在 7x7 中的放置范围为 0..4，保证 3x3 mask 不越界。
#    - 已知 minority 子集仍通过唯一候选筛选得到，保证 solver 的 placement assert 可满足。
#    - overlay 只写入目标单元格的 3x3 interior，不触碰 1 像素分隔线，也不会跨单元格重叠。
# e) 视觉可观察性
#    - 训练样例中，输入仍显示少量 minority 锚点；输出会把整块 3x3 放大区域用统一描边颜色标出来。
#    - 颜色规则是可见的固定样式：新出现的 overlay 颜色总是表示“被放大图案覆盖到的单元格边界”。
# f) 可视化整洁度
#    - 叠加层只占每个 3x3 单元格的 8 个边框像素，中心像素保留，因此底图仍可读。
#    - 不在分隔线作画，避免轮廓与网格线粘连成大片噪声。
# g) 颜色变化可推断性
#    - 输入颜色只从 1..9 中采样，overlay 固定为 0，因此输出新增颜色只有一种，语义稳定。
#    - 没有使用颜色 id 进行排序或条件分支；颜色变化只由“是否属于放大掩码前景”这个几何条件驱动。
# h) 新的隐式假设
#    - main() 假设 overlay 颜色不会出现在输入里，否则输入单元格模式统计可能被污染；generator 保证这一点。
#    - main() 假设单元格 interior 至少 3x3，才能画出“边框 + 中心保留”；本题固定 inner=3，generator 维持原规格。


OVERLAY_COLOR = 0


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
    """Read every cell interior and cache each distinct flattened pattern."""
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


def _infer_minority_mask(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Recover the minority-cell mask and its unique placement in the cell grid."""
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    patterns, cell_pat = _extract_cell_patterns(g, step)

    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            p = cell_pat[i][j]
            counts[p] = counts.get(p, 0) + 1
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns (minority + majority)"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat

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
    assert len(candidates) == 1, "must have a unique mask placement aligning existing minority cells"
    top, left = candidates[0]
    return mask, existing, step, top, left


def _cell_frame_mask(inner: int) -> np.ndarray:
    """Border pixels of a cell interior; the center area remains transparent."""
    frame = np.ones((inner, inner), dtype=bool)
    if inner > 2:
        frame[1:-1, 1:-1] = False
    return frame


def _apply_overlay(g: np.ndarray, mask: np.ndarray, step: int, top: int, left: int) -> np.ndarray:
    """Overlay a border-only marker on every active cell-sized stamp."""
    inner = step - 1
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
    Lift a minority cell's inner stencil to the cell-grid scale, then render it as a translucent-style outline.

    Input structure:
    - A square grid is partitioned into equal 3x3 cell interiors by single-color divider lines.
    - Nearly every cell repeats one majority interior pattern, while a small subset shows one minority pattern.

    Rule:
    - Read the minority cell as a 3x3 activation stencil, using its dominant color as the foreground mask.
    - Find the unique placement of that stencil on the 7x7 cell lattice so the given minority cells fit inside it.
    - For every active stencil position, keep the underlying cell content but recolor only that cell's interior border as an overlay mark.

    Key insight:
    The odd cell still encodes the large-scale fractal layout, but the final composition is no longer a hard replacement:
    the enlarged foreground is shown through a separate outline layer placed on top of the original grid.
    """
    g = np.asarray(grid, dtype=int)
    mask, _, step, top, left = _infer_minority_mask(g)
    return _apply_overlay(g, mask, step, top, left)


# --------------- generator (o5x) ---------------

# --- CFG: tunable generation parameters ---
CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
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


def _precompute_masks():
    """Enumerate all valid LR-symmetric 3x3 masks, keyed by active pixel count."""
    weights = [len(g) for g in _LR_GROUPS]
    by_count = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks = []
        for k in range(1, len(_LR_GROUPS) + 1):
            for combo in combinations(range(len(_LR_GROUPS)), k):
                if sum(weights[i] for i in combo) != inactive_count:
                    continue
                mask = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
                for i in combo:
                    for r, c in _LR_GROUPS[i]:
                        mask[r, c] = False
                if mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform LR-symmetric 3x3 pattern using two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask() -> np.ndarray:
    """Pick a random valid minority mask."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _placement_candidates(mask: np.ndarray, subset, top: int, left: int) -> list[tuple[int, int]]:
    """Find all (t, l) offsets where the given subset of active cells is consistent with mask."""
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


def _choose_given_subset(mask: np.ndarray, top: int, left: int):
    """Choose a subset of active mask positions that uniquely determines the placement."""
    active = [tuple(coord) for coord in np.argwhere(mask)]
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


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, minority: np.ndarray, active_cells) -> np.ndarray:
    """Construct a full grid from majority cells plus a selected set of minority cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _expected_output(inp: np.ndarray, mask: np.ndarray, top: int, left: int) -> np.ndarray:
    """Apply the known overlay rule from the generation parameters."""
    return _apply_overlay(inp, mask, CELL_STEP, top, left)


def _gen_attempt():
    """Single generation attempt: build input, derive the expected overlay output, verify with solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(INPUT_COLORS, 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, minority, given_cells)

    expected = _expected_output(inp, mask, top, left)
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
    raise RuntimeError("failed to generate a valid b99e7126 D16 input")
