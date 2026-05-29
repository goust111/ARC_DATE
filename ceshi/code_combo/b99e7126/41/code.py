# Combo variant of b99e7126 (index 41)
# combo_spec: D3/0+D5/2
# Name: 擦除式叠加 + 居中放置

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random
from arc_utils import paste

# Design analysis
# combo_spec:
#   identity:
#     base_task: b99e7126
#     fused_dims:
#       - "D3/0: 放大后的 3x3 单元格级图案不再根据 clue 的源偏移落位，而是固定居中到整个 7x7 单元格网格。"
#       - "D5/2: 放大图案的活跃单元不再写成 minority 纹理，而是整块擦成分隔线背景色。"
#   concretization:
#     scene:
#       - "输入仍是 29x29 像素网格，内部是 7x7 个 3x3 单元格，中间隔着 1 像素分隔线。"
#       - "多数单元格仍共享同一个 majority 纹理；少数 3-4 个 minority clue 单元格仍共享另一种 3x3 纹理。"
#       - "minority 纹理内部高频色定义 3x3 掩码的活跃像素，这个掩码仍被理解为要放大到单元格级的分形蓝图。"
#       - "但输出不再沿 clue 对应的源偏移去扩展，而是把整张 3x3 蓝图放到 7x7 网格中心；同时活跃位置被擦成分隔线色的实心洞块。"
#     traced_example:
#       - "设 minority 掩码在右上、中央、左下等 6-7 个像素为活跃。原题会让 clue 所在的那块 3x3 单元格区域作为锚点，并把其余活跃位置补成 minority 纹理。"
#       - "本融合版仍允许输入 clue 出现在某个非中心源偏移，例如左上 3x3 单元格区；solver 先确认这些 clue 确实能唯一对应一个源偏移。"
#       - "真正生成输出时，不使用这个源偏移做落位，而是把同一个 3x3 掩码挪到全局中心 `(2, 2)`；落在活跃位的整块 3x3 单元格内部被擦成背景色。"
#       - "因此差异同时发生在两步：位置从局部锚定改成全局居中，覆盖方式从 minority 贴花改成背景挖空。"
#   complexity_calibration:
#     - "保留原题的 7x7 单元格、3x3 掩码、6-7 个活跃像素、3-4 个 clue 子集，保证题目仍然需要识别异常纹理并理解其分形含义。"
#     - "generator 强制 clue 源偏移优先选择与中心有曼哈顿距离至少 2 的位置，避免大多数样例退化成“本来就几乎在中心”。"
#     - "擦除颜色固定为 visibly present 的分隔线色，而不是新颜色；这让‘中央挖洞’在人眼上稳定可见。"
#   implicit_assumption_audit:
#     - "`_infer_cell_grid_params` 只依赖整行整列分隔线；擦除只发生在单元格内部，所以不会破坏该假设。"
#     - "原题里 `counts` 的最小/最大频次用于区分 minority 与 majority；融合版输入仍只保留这两类纹理，因此该推断仍成立。"
#     - "原题里 `active_color = argmax(...)` 假定 minority 前景色在 3x3 内占多数；生成器继续限制活跃像素数为 6 或 7。"
#     - "原题默认输出可以从输入副本局部覆写得到；这里该假设失效，因为非中心 clue 在输出里必须消失，所以 solver 改为先重建一张全 majority 底板。"
#     - "原题的唯一候选 `(top, left)` 会直接决定输出落位；这里它只作为输入合法性的验证信息保留，真正的目标位置改为由网格几何中心唯一决定。"
#     - "新假设是 `n - mask_size` 在行列上都为偶数，才能整数居中；本任务固定 `n=7, mask_size=3`，生成器始终满足。"
#   placement_legality:
#     - "输出中的中心放置使用 `(n - mh) // 2, (n - mw) // 2`，对 7x7 中的 3x3 掩码恰好得到 `(2, 2)`，不会越界。"
#     - "输入 clue 仍从某个合法源偏移 `0..4` 的掩码活跃位中选 3 或 4 个，且要求这一子集唯一决定该源偏移。"
#     - "generator 排除中心源偏移，确保 D3/0 的居中控制变量真正影响输出；必要时重试直到找到合法 clue 子集。"
#     - "输出的擦除块只作用于中心 3x3 掩码覆盖到的单元格内部，不会跨出单元格边界，也不会破坏外层分隔线。"
#   visual_observability:
#     - "解题者在输入里会看到少量非中心 minority clue，输出里这些 clue 自身不保留，而是在网格中央出现同形状的洞。"
#     - "多组样例中 clue 源偏移会变化，但中央洞始终固定在中间，这能让人观察到‘位置被重置为全局居中’而不是按 clue 原地扩展。"
#     - "洞块与分隔线同色，因此人眼会把它读成被挖空的背景区域，而不是第三种纹理。"
#   cleanliness:
#     - "输出只包含 majority 铺贴、分隔线以及中央洞块三种视觉角色；不会出现中心洞和旧 clue 同时并存导致的混乱。"
#     - "由于先重建底板再擦除，所有非中心 clue 都被干净移除，不会在外围留下多余 minority 残影。"
#   fusion_metric_plan:
#     - "量化 D3 是否生效：恢复输入的唯一源偏移，与固定中心偏移比较，统计其曼哈顿距离；若始终大于 0，则融合没有退化成锚定放置。"
#     - "量化融合输出与‘仅擦除、不居中’版本的差异：比较两者像素级差异数；若始终大于 0，则居中控制变量确实改变了最终图像。"


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


def _cell_origin(i: int, j: int, step: int) -> tuple[int, int]:
    return i * step + 1, j * step + 1


def _extract_cell_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    inner = step - 1
    n = (g.shape[0] - 1) // step
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0, c0 = _cell_origin(i, j, step)
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(pat_arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)
    return patterns, cell_pat


def _count_patterns(cell_pat: list[list[tuple[int, ...]]]) -> dict[tuple[int, ...], int]:
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    return counts


def _existing_minority_mask(cell_pat: list[list[tuple[int, ...]]], minority_pat: tuple[int, ...]) -> np.ndarray:
    n = len(cell_pat)
    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat
    return existing


def _placement_candidates(mask: np.ndarray, existing: np.ndarray) -> list[tuple[int, int]]:
    n = existing.shape[0]
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
    return candidates


def _centered_offset(n: int, mh: int, mw: int) -> tuple[int, int]:
    assert n >= mh and n >= mw, "mask must fit in the cell grid"
    assert (n - mh) % 2 == 0 and (n - mw) % 2 == 0, "centering requires even slack"
    return (n - mh) // 2, (n - mw) // 2


def _build_majority_grid(separator: int, majority: np.ndarray, n: int, step: int) -> np.ndarray:
    grid = np.full((n * step + 1, n * step + 1), separator, dtype=int)
    for i in range(n):
        for j in range(n):
            paste(grid, majority, *_cell_origin(i, j, step))
    return grid


def _build_erased_output(separator: int, majority: np.ndarray, mask: np.ndarray, top: int, left: int, n: int, step: int) -> np.ndarray:
    inner = step - 1
    erased = np.full((inner, inner), separator, dtype=int)
    out = _build_majority_grid(separator, majority, n, step)
    for r, c in np.argwhere(mask):
        paste(out, erased, *_cell_origin(top + int(r), left + int(c), step))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the minority cell as a fractal stencil, recenter that stencil on the whole lattice, and carve it out.

    Input structure:
    - The picture is a regular 7x7 cell lattice of 3x3 interiors separated by a uniform divider color.
    - Almost every cell shows one majority texture, while a small clue subset shows a different minority texture.
    - The minority texture itself is a 3x3 two-color motif whose dominant color marks active pixels.

    Rule:
    - Interpret the minority texture as a binary 3x3 mask and verify that the visible minority clues are consistent with one unique source placement of that mask on the 7x7 cell grid.
    - Ignore that source placement when drawing the answer: instead, place the full 3x3 mask at the global center of the cell lattice.
    - Rebuild the grid as majority texture everywhere, then erase every center-mask active cell interior to the divider color.

    Key insight: The minority texture still provides the fractal shape, but the clue cells no longer anchor the final location; they only reveal the stencil, which is then recentered and used as a carving mask.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    bg = int(g[0, 0])

    patterns, cell_pat = _extract_cell_patterns(g, step)
    counts = _count_patterns(cell_pat)
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns (minority clues + majority)"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    mh, mw = mask.shape

    existing = _existing_minority_mask(cell_pat, minority_pat)
    source_candidates = _placement_candidates(mask, existing)
    assert len(source_candidates) == 1, "minority clues must define a unique source placement"

    center_top, center_left = _centered_offset(n, mh, mw)
    return _build_erased_output(bg, majority_arr, mask, center_top, center_left, n, step)


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
CENTER_TOP = (CELL_GRID_N - MASK_SIZE) // 2
CENTER_LEFT = (CELL_GRID_N - MASK_SIZE) // 2

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _precompute_masks() -> dict[int, list[np.ndarray]]:
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


def _placement_candidates_from_subset(mask: np.ndarray, subset, top: int, left: int) -> list[tuple[int, int]]:
    """Find all source offsets consistent with a subset of active relative coords."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True
    return _placement_candidates(mask, existing)


def _choose_given_subset(mask: np.ndarray, top: int, left: int):
    """Choose a clue subset that uniquely determines the source placement."""
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(active, size)
            if _placement_candidates_from_subset(mask, subset, top, left) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _source_offsets() -> list[tuple[int, int]]:
    """Prefer non-center source offsets that visibly differ from the forced centered output."""
    far: list[tuple[int, int]] = []
    near: list[tuple[int, int]] = []
    for top in range(PLACEMENT_RANGE):
        for left in range(PLACEMENT_RANGE):
            if (top, left) == (CENTER_TOP, CENTER_LEFT):
                continue
            pos = (top, left)
            if abs(top - CENTER_TOP) + abs(left - CENTER_LEFT) >= 2:
                far.append(pos)
            else:
                near.append(pos)
    random.shuffle(far)
    random.shuffle(near)
    return far + near


def _build_input(separator: int, majority: np.ndarray, minority: np.ndarray, minority_cells) -> np.ndarray:
    """Construct an input grid from a majority base plus selected minority clue cells."""
    grid = _build_majority_grid(separator, majority, CELL_GRID_N, CELL_STEP)
    for i, j in minority_cells:
        paste(grid, minority, *_cell_origin(int(i), int(j), CELL_STEP))
    return grid


def _gen_attempt():
    """Single generation attempt: build centered erased output, then synthesize off-center clues."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    out = _build_erased_output(separator, majority, mask, CENTER_TOP, CENTER_LEFT, CELL_GRID_N, CELL_STEP)

    for top, left in _source_offsets():
        subset = _choose_given_subset(mask, top, left)
        if subset is None:
            continue
        given_cells = [(top + int(r), left + int(c)) for r, c in subset]
        inp = _build_input(separator, majority, minority, given_cells)

        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    return None


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
