# Variant of d59b0160
# Dimension: D14 — 网格输出形状 (grid_shape)
# Change: 输出与输入尺寸完全相同（same），删除的分量区域变为背景色... → 输出仅包含被删除分量的区域

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis
# a) 具体化与行为追踪
#    - 原题的关键识别机制不变：左上角 4x4 图例给出 3 个目标色，扫描所有非背景 4-连通分量，
#      若某分量包含全部目标色，则该分量属于“应删除对象”。
#    - D14 改动落在“输出形状”而非“匹配逻辑”：不再返回整张 16x16 擦除后的场景，而是只返回
#      应删除对象所在的区域。为了让变化可观察且不退化成一块纯背景，采用“提取”版本：
#      输出为匹配分量删除前的原图案，并裁成其最小外接矩形。
#    - 具体小例子：
#      原逻辑：某个 5x3 矩形分量含有图例中的 3 个颜色 -> 在 16x16 输出里把这 5x3 区域改成 bg。
#      新逻辑：同一个 5x3 分量仍被识别为匹配对象 -> 输出直接变成这个 5x3 图案本身。
#      差异：识别步骤相同，变化发生在“命中后如何构造输出”。
# b) 复杂度校准
#    - 若允许 2-3 个删除分量同时存在，输出会变成多个对象的联合裁剪框，例子少时不易判断
#      “是提取单个对象还是提取一组对象”。因此生成器固定为“恰好 1 个匹配分量”，
#      solver 仍支持多个匹配分量的联合裁剪，保证规则本身完整，但样例保持清晰。
#    - 其余仍保留 3-5 个干扰分量，避免任务退化成只看唯一对象。
# c) 隐式假设审查
#    - `_bg_boundary` 仍依赖外边框大多为背景色；生成器继续让外围主要为 7，成立。
#    - `_find_legend` 仍依赖左上角 bottom/right 单色边框；图例构造保持不变。
#    - `components_mask(g != bg)` 仍把黑色 0 视为对象内部而非背景；生成器中的对象仍是 0 底色矩形。
#    - 新增假设是“至少存在一个匹配分量”；生成器固定放置 1 个删除分量保证这一点。
#    - 新输出使用 bbox 裁剪，因此假设匹配分量的 bbox 不会覆盖整张图；图例占左上角且对象之间有间隔，
#      组件尺寸也远小于整张 16x16，成立。
# d) 放置合法性
#    - 仍使用 `random_free_location(..., padding=1)`，保证分量与图例、分量与分量之间至少隔 1 个背景格，
#      从而 4-连通分量不会意外合并。
#    - 删除分量与保留分量都以整块矩形 patch 放置，永不越界。
# e) 视觉可观察性
#    - 输出尺寸从固定 16x16 改成对象自身 bbox，肉眼能直接看出“只剩被筛中的对象”。
#    - 输出对象保留删除前图案，而不是输出纯背景空框，避免 D14 变化不可见。
# f) 可视化整洁度
#    - 图例仍在左上角，主体对象彼此分离；输出只含一个对象，信息密度比原题更低、更干净。
# g) 颜色变化可推断性
#    - 不引入任何新的颜色映射；颜色仍只承担“图例目标色是否齐全”的作用。
# h) 变体量化指标
#    - 关键指标 1：`main(input).shape != input.shape`，说明 D14 的裁剪行为确实发生。
#    - 关键指标 2：输出必须含有全部图例目标色，说明是“提取被删除对象”，不是裁空背景。
# i) `main()` 的新假设与 generator 对应保证
#    - `main()` 对每个匹配分量都保留原像素并在联合 bbox 中裁剪；generator 通过“唯一匹配分量”
#      让训练样例无歧义。
#    - `main()` 默认 bbox 内非匹配位置填 bg；generator 让匹配分量占满自身 bbox，因此输出不会出现
#      无意义的大块背景。


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the top-left legend whose bottom edge and right edge share one border color.
    Return (size, border_color, interior_target_colors).
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop out the component(s) specified by the legend palette.

    Input structure:
    - Most cells are the background color, and the top-left corner holds a small legend box whose bottom and right edges share one border color.
    - The legend interior shows the target color set.
    - Away from the legend, separated 4-connected multicolor components sit on the background.

    Rule:
    - Read the target colors from inside the legend.
    - Find every non-background connected component that contains all of those colors.
    - Compute the minimum bounding rectangle covering the matching component(s), and output only that crop.
      Inside the crop, keep pixels that belong to matching component(s) and set every other cell to background.

    Key insight: The legend still identifies which component would be erased in the original task, but this variant extracts that matched object region instead of returning the whole scene with holes punched out.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _border, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    matches: list[np.ndarray] = []
    for pts in components_mask(g != bg, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            matches.append(pts)

    assert matches, "no connected component matches the legend palette"

    all_pts = np.concatenate(matches, axis=0)
    r0, c0, r1, c1 = bbox_coords(all_pts)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for pts in matches:
        for r, c in pts:
            out[int(r) - r0, int(c) - c0] = int(g[r, c])
    return out


# --------------- generator (D14 variant) ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
TALL_H = (8, 12)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

# -- Component fill density --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 8]
NUM_DELETED = 1  # unique extracted target for visual clarity

# -- Generation retry budget --
MAX_RETRIES = 400


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box at top-left and place target colors."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    # Keep one target disconnected from the border so the legend itself never matches all 3 colors.
    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    border_side = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        p for p in border_side if abs(p[0] - isolated[0]) + abs(p[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color
        out[r, c] = color


def _component_dims() -> tuple[int, int]:
    """Sample component dimensions according to shape-mode weights."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], deleted: bool) -> np.ndarray:
    """Build an h x w solid non-bg component with color-0 body and sprinkled colors."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if deleted:
        lo = max(lo, NUM_TARGETS + 1)
        hi = max(hi, lo)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if deleted:
        colors = list(targets) + [random.choice(PALETTE) for _ in range(n_colored - NUM_TARGETS)]
    else:
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        allowed = non_targets + random.sample(targets, used_target_count)
        colors = [random.choice(allowed) for _ in range(n_colored)]

    random.shuffle(colors)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose output is the unique deleted component crop."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * NUM_DELETED + [False] * (component_count - NUM_DELETED)
    random.shuffle(roles)

    expected_out = None
    for deleted in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, deleted)
        try:
            r, c = random_free_location(
                inp,
                np.zeros_like(patch),
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(inp, patch, r, c)
        if deleted:
            expected_out = patch.copy()

    if expected_out is None:
        return None
    if np.array_equal(main(inp), expected_out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
