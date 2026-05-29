# Combo variant of b99e7126 (index 8)
# combo_spec: D5/2+D16/2
# Name: 擦除式叠加 + XOR式叠加

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random
from arc_utils import paste

# Design analysis
# a) 具体化与行为追踪:
#    原题先识别 minority 单元格，把它内部 3x3 的高频色当作蓝图活跃位，再把这张 3x3 蓝图
#    放大到 7x7 单元格网格上；原题活跃位统一写成 minority。D5 把“写成 minority”改成“擦成
#    separator 背景洞块”。D16 把“统一写入”改成“在活跃位做 XOR 翻转”。融合后不能退化成
#    先擦除再覆盖，而应把 XOR 的两端状态定义为“minority clue 纹理”和“separator 洞块”：
#    若活跃位当前已是 minority clue，则翻转后擦成洞块；若活跃位当前还是 majority，则翻转后
#    写成 minority。例: 活跃位共 6 个、输入给出 3 个 minority clue，则原题输出是 6 个
#    minority；D5 输出是 6 个洞块；D16 输出是 3 个 minority + 3 个 majority 互换；本融合
#    输出是 3 个洞块 + 3 个新 minority，既有“消失”也有“出现”。
# b) 复杂度校准:
#    保留原题的 7x7 单元格、3x3 内纹理、3x3 蓝图、6-7 个活跃位、3-4 个 clue，定位难度不变。
#    输出同时出现 minority 与 separator 洞块，视觉上明显区别于原题与单变体，但仍只使用少量
#    稳定角色，避免再引入新纹理类别。
# c) 隐式假设审查:
#    - `_infer_cell_grid_params` 只依赖完整分隔线；洞块只改单元格内部，不破坏此假设。
#    - `counts` 推断输入只有 majority 与 minority 两类单元格；生成器继续保证输入只有这两类。
#    - `active_color = argmax(...)` 依赖 minority 内高频色确实是蓝图前景；生成器继续让活跃像素
#      数为 6 或 7，因此高频色稳定。
#    - 原最终循环假设“活跃位无条件覆写成一个固定模式”；融合后这一假设失效，必须读取输入该位
#      当前是否是 minority clue，再决定写洞块还是写 minority。
#    - 融合后的新假设是：输入里所有 minority clue 都落在隐藏蓝图活跃位上；生成器用 subset
#      仅从 active mask 中选 clue，并验证唯一摆放。
# d) 放置合法性:
#    蓝图仍是 3x3，偏移范围仍为 0..4，保证完整落在 7x7 单元格网格内。输入 clue 仍是活跃位
#    的真子集，因此不存在 clue 越界或落在 inactive 位上的情况。输出只在合法单元格内部写入
#    minority 或洞块，不发生对象重叠。
# e) 视觉可观察性:
#    解题者先从少数异常单元格识别出蓝图，再看到输出里“给出的 clue 被抹掉，而没给出的活跃位
#    被点亮成 minority”。这种同一区域内的出现/消失共存，能从 2-4 组样例中稳定看出不是简单
#    completion，也不是纯擦除。
# f) 可视化整洁度:
#    单元格内部始终只会是 majority、minority、或纯 separator 洞块三种整块纹理，不会出现
#    局部混色叠加。分隔线颜色与洞块一致，能自然读成“被挖空”的状态。
# g) 变化的可推断性:
#    颜色规则不依赖颜色编号，只依赖可见角色：分隔线色 = 洞块背景；minority 纹理 = 前景 clue。
#    majority 只是初始底板，活跃位的条件变化可由结构位置推断，不需要记忆抽象颜色映射。
# h) 防退化策略:
#    生成器强制 clue 数量是活跃位真子集，因此每个样例同时包含“minority->洞块”和
#    “majority->minority”两类翻转；这样不会退化为原题、纯擦除、或纯 D16 XOR。
# i) main() 的新假设与生成器保障:
#    `main()` 现在假设活跃位只有两种需要写出的结果：已有 clue 的位置写洞块，未给 clue 的活跃
#    位置写 minority，inactive 位置恢复 majority。生成器显式按此构造目标输出，并用
#    `np.array_equal(main(inp), out)` 验证。


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden fractal stencil and use it to flip active cells between visible clue ink and carved-out holes.

    Input structure:
    - The board is split by one-color divider lines into a 7x7 lattice of equal 3x3 cell interiors.
    - Almost all cells share one majority texture, while a smaller clue set shows one minority texture.

    Rule:
    - Read the minority cell's 3x3 interior as a binary stencil, taking its dominant color as the active pixels.
    - Place that 3x3 stencil at the unique position on the cell lattice that contains every visible minority clue.
    - For each stencil-active cell, apply the fused XOR/erase rule: an already-minority clue cell is erased to the divider color, while an active cell that was still majority becomes the minority texture. Stencil-inactive cells remain majority.

    Key insight:
    - The odd cell is still a miniature blueprint for a larger 3x3 placement, but the enlarged placement is no longer a plain completion. It behaves like a toggle mask whose "off" state is a carved background hole, so the given clues disappear and the missing active positions light up.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1
    bg = int(g[0, 0])

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

    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            p = cell_pat[i][j]
            counts[p] = counts.get(p, 0) + 1
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns (minority clues + majority)"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

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
    assert len(candidates) == 1, "must have a unique mask placement aligning existing minority clues"
    top, left = candidates[0]

    erased_cell = np.full((inner, inner), bg, dtype=int)
    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            if mask[i, j]:
                target_pat = erased_cell if existing[top + i, left + j] else minority_arr
            else:
                target_pat = majority_arr
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


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
COLOR_POOL = 10
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
    weights = [len(group) for group in _LR_GROUPS]
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
        for k in range(1, len(_LR_GROUPS) + 1):
            for combo in combinations(range(len(_LR_GROUPS)), k):
                if sum(weights[idx] for idx in combo) != inactive_count:
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


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a random LR-symmetric 3x3 pattern using two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(value == choices[0] for value in choices):
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
    """Find all offsets where the given subset of clue cells is consistent with the mask."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True

    candidates: list[tuple[int, int]] = []
    for cand_top in range(PLACEMENT_RANGE):
        for cand_left in range(PLACEMENT_RANGE):
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[cand_top : cand_top + MASK_SIZE, cand_left : cand_left + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[cand_top : cand_top + MASK_SIZE, cand_left : cand_left + MASK_SIZE] & ~mask):
                continue
            candidates.append((cand_top, cand_left))
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


def _build_grid(separator: int, majority: np.ndarray, minority: np.ndarray, minority_cells) -> np.ndarray:
    """Construct a full grid with majority cells everywhere except the chosen minority clue cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in minority_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _build_fusion_output(
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    active_cells,
    given_cells,
) -> np.ndarray:
    """Construct the fused output: given clue cells become holes; other active cells become minority."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    erased = np.full((CELL_INNER, CELL_INNER), separator, dtype=int)
    given_set = set(given_cells)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        patch = erased if (i, j) in given_set else minority
        paste(grid, patch, *_cell_origin(i, j))
    return grid


def _gen_attempt():
    """Single generation attempt: build fused target output, derive clues, and verify."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + int(r), left + int(c)) for r, c in subset]

    inp = _build_grid(separator, majority, minority, given_cells)
    out = _build_fusion_output(separator, majority, minority, active_cells, given_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 combo D5/2+D16/2 input")
