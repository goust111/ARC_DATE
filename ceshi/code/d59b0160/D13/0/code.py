# Variant of d59b0160
# Dimension: D13 — 保留分量的处理方式 (transform)
# Change: 不匹配的连通分量完全保留不变（原样复制到输出）... → 保留分量被重着色为统一颜色

import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis (pre-implementation)
# spec:
#   unchanged_core:
#     - 背景仍由外边框多数色定义，固定为 7。
#     - 左上角仍有 4x4 图例，底边与右边为绿色 3，内部 3 个单像素给出目标颜色集合。
#     - 其余对象仍是放在背景上的 4 连通多色矩形分量，彼此至少隔 1 个背景格。
#   D13_change:
#     - 原题: 含全体图例色的分量删除，其余分量原样保留。
#     - 变体: 含全体图例色的分量仍删除；其余分量不再保留内部花纹，而是整块改涂成该分量内部出现次数最多的颜色。
#   concretization_and_trace:
#     - 例: 图例色为 {1,4,8}。某矩形分量里颜色频次为 6:10 格, 0:4 格, 4:2 格，因此它缺少 1 和 8，不会被删。
#     - 原逻辑输出: 这个分量 16 格全部照抄。
#     - 新逻辑输出: 这 16 格仍保留同一矩形轮廓，但全部改成 6。
#     - 另一分量若同时含有 1/4/8，则两版都会把它整块涂成背景 7。
#   complexity_calibration:
#     - 只改 D13，不增加第二套图例或额外排序。
#     - 为了让“重着色”肉眼可见，保留分量不能在输入中已经单色，也不能频次并列。
#     - 因此 generator 强制每个保留分量有唯一多数色且至少有一种少数色，删除分量也维持同类外观，避免角色靠风格泄漏。
#   implicit_assumptions_review:
#     - main() 仍用 components_mask(g != bg)；这依赖对象内部所有颜色都不是背景。生成器继续用实心矩形满足此点。
#     - 原题对未匹配分量直接不写 out，因此图例内部像素即使彼此 4 连通也无害。
#     - 新题若无条件重着色全部未删分量，会把说明区本身染平；因此 main() 需要把左上图例方块视为说明区并跳过。
#     - 最常见色若出现并列会引入隐藏 tie-break。生成器保证非删除矩形的多数色唯一，从而不把“取最小颜色”变成题意。
#   placement_legality:
#     - 组件通过 random_free_location 放置，patch 全部视为占用，padding=1 保证对象之间和图例之间都不会 4 连通或 8 邻接粘连。
#     - 尺寸下界调到至少 3x3，使“唯一多数色 + 至少 3 个目标色”可以同时成立，不会出现构造时无解。
#   visual_observability:
#     - 训练样例中会同时出现两类变化: 有的分量整块消失，有的分量保留轮廓但内部纹理被抹平。
#     - 统一色来自对象自身的显著多数色，而不是颜色编号排序或固定外部映射。
#   cleanliness_and_color_inference:
#     - 输出不会叠加新标记，只是删块或整块改单色，视觉上比原题更干净。
#     - 颜色种类仍限制在原题对象色集合 {0,1,2,4,5,6,8,9} 内，且单个保留分量只有一个主色需要被读出，推断负担可控。


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Find the smallest top-left square whose bottom edge and right edge share one color.
    Its interior colors, excluding bg and the border color, define the target set.
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(x)
            return int(sz), int(x), colors
    assert False, "no legend border found in grid"


def _mode_color(values: np.ndarray) -> int:
    """Deterministic mode; on ties np.unique ordering makes the smallest color win."""
    vals, cnts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[cnts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete matching components and flatten the survivors into one color.

    Input structure:
    - Background fills most of the board.
    - A small top-left legend is framed on its bottom and right edges by one border color; the interior lists the target colors.
    - Elsewhere sit 4-connected multicolor components separated by background.

    Rule:
    - Read the target color set from the legend interior.
    - For each non-background connected component outside the legend square, test whether it contains all legend colors.
    - Components that contain the full set are erased to background; every other non-legend component is kept but repainted to its own most frequent color.
    - Output size stays unchanged.

    Key insight: The legend still decides which objects disappear, but the surviving objects no longer preserve their internal pattern; only their silhouette and dominant color remain.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, _border, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        rows, cols = pts[:, 0], pts[:, 1]
        if bool(np.all(rows < sz) and np.all(cols < sz)):
            continue
        comp_vals = g[rows, cols]
        comp_colors = set(map(int, comp_vals))
        if legend_colors.issubset(comp_colors):
            out[rows, cols] = bg
        else:
            out[rows, cols] = _mode_color(comp_vals)
    return out


# --------------- generator (output-first with dominant-color recolor) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
OBJECT_COLORS = [0] + PALETTE
NUM_TARGETS = 3

SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
TALL_H = (5, 10)
TALL_W = (2, 3)
WIDE_H = (3, 4)
WIDE_W = (4, 7)
MED_H = (3, 6)
MED_W = (3, 6)

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [3, 6, 5]
NUM_KEPT = 3

MAX_RETRIES = 400


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend at top-left; its components are already monochrome."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    border_side = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        p for p in border_side
        if abs(p[0] - isolated[0]) + abs(p[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color
        out[r, c] = color


def _component_dims() -> tuple[int, int]:
    """Sample component dimensions while keeping area large enough for a unique mode."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], deleted: bool) -> tuple[np.ndarray, int]:
    """
    Build a solid rectangle with one guaranteed dominant color and several minority cells.
    The dominant color is the recolor target for non-deleted components.
    """
    area = h * w
    non_targets = [v for v in OBJECT_COLORS if v not in targets]

    if deleted:
        allowed = list(OBJECT_COLORS)
    else:
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        allowed_targets = random.sample(targets, used_target_count)
        allowed = non_targets + allowed_targets

    dominant = random.choice(allowed)
    pool_without_dominant = [v for v in allowed if v != dominant]
    if not pool_without_dominant:
        raise ValueError("component needs at least one minority color")

    if deleted:
        required = [t for t in targets if t != dominant]
        if not required:
            required = [random.choice(pool_without_dominant)]
    else:
        required = [random.choice(pool_without_dominant)]

    max_mutations = (area - 1) // 2
    min_mutations = len(required)
    if min_mutations > max_mutations:
        raise ValueError("component too small for a unique dominant color")
    n_mutations = random.randint(min_mutations, max_mutations)

    patch = np.full((h, w), dominant, dtype=int)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_mutations)

    extra_pool = [v for v in pool_without_dominant]
    mutation_colors = list(required)
    mutation_colors.extend(random.choice(extra_pool) for _ in range(n_mutations - len(required)))
    random.shuffle(mutation_colors)

    for (r, c), color in zip(coords, mutation_colors):
        patch[r, c] = color
    return patch, dominant


def _gen_attempt() -> np.ndarray | None:
    """Construct a valid input/output pair and return the input on success."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for deleted in roles:
        h, w = _component_dims()
        try:
            patch, dominant = _component_patch(h, w, targets, deleted)
            r, c = random_free_location(
                inp, patch, transparent=BG, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(inp, patch, r, c)
        if not deleted:
            paste(out, np.full_like(patch, dominant), r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
