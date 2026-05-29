# Variant of d59b0160
# Dimension: D6 — 删除操作的具体实现 (deletion)
# Change: 匹配的连通分量的所有像素被替换为背景色(7/橙色)... → 删除后剩余像素重新着色为分量中最多的非图例颜色

# Design analysis
# design_analysis:
#   concrete_behavior:
#   - 输入仍然是 16x16、背景色 7、左上角 4x4 绿色边框图例、其余位置放若干 4-连通矩形分量。
#   - 图例内部 3 个颜色给出目标色集合。组件判定条件不变：只要某个非背景连通分量同时含有全部 3 个目标色，就触发“删除”。
#   - 但这里的“删除”不再是整块清空。命中组件里，目标色像素会被删成背景色 7；其余像素保留原位置，然后统一改成该组件里出现次数最多的非目标色。
#   - 因此输出中的命中组件会保留大致外轮廓，只是在原目标色位置留下若干洞，并且剩余像素被同色化；未命中组件和图例保持原样。
#   flow_trace_example:
#   - 例：图例目标色 = {1,4,9}。某个矩形组件里有很多 6，少量 0 和 8，以及若干 1/4/9。
#   - 原题：因为它包含 {1,4,9}，整个组件所有像素都变成 7。
#   - 变体：先把 1/4/9 的那些像素删成 7，再把剩余的 0/6/8 全部改成其中最多的颜色 6。
#   - 差异：原题输出是整块消失；现在输出是“同色实体 + 几个背景孔洞”。
#   complexity_calibration:
#   - 若命中组件总是以 0 为多数，人类容易误判为“删掉目标色后统一刷黑”。因此生成器让命中组件多数色通常来自非目标调色板中的单一颜色，并把该多数做成显著唯一多数。
#   - 同时仍保留少量 0 或其他非目标色，保证确实存在“重着色”步骤，而不是输入本来就近乎单色。
#   implicit_assumptions_review:
#   - `_bg_boundary` 仍依赖外框主要是背景色，这在生成器中保持不变。
#   - `_find_legend` 仍依赖左上角 legend 的下边和右边同色；legend 布局不改。
#   - 旧版 `main()` 对命中组件直接整块赋背景色；这一写法依赖“删除 == 全块消失”。变体下该前提失效，必须改成逐像素区分目标色与非目标色，再基于残留像素统计多数色。
#   - 命中组件如果删完目标色后没有残留像素，重着色步骤将无事可做；代码允许这种情况，但生成器始终构造有残留像素的矩形分量。
#   placement_validity:
#   - 组件仍按整块 bbox 放置，使用 `random_free_location(..., padding=1)` 保证彼此和 legend 至少隔 1 个背景格，不重叠、不越界。
#   - 即使输出里命中组件内部出现背景孔洞，也不会影响 solver 对输入的连通分量识别，因为输入阶段这些孔洞位置仍是非背景像素。
#   visual_observability:
#   - 输出中的命中组件不会完全消失，而是保留轮廓并出现与图例色对应的孔洞，这与原题肉眼差异明显。
#   - 多数色由组件内部可见频率决定，不依赖颜色 id 排序；若遇到理论上的并列，solver 用扫描顺序首次达到最大值的颜色打破并列，生成器则避免真正并列。
#   cleanliness_and_color_inference:
#   - 目标色只出现在 legend 和被打孔的位置，空间角色清晰。
#   - 命中组件的输出通常呈单一颜色大块，有助于从 2-4 组样例中归纳“删掉图例色后，剩下的都变成该组件里最多的那种颜色”。

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border (bg defined by the frame, not global frequency)."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    The legend is a top-left corner region whose bottom edge and right edge are the same color x.
    We detect the smallest size sz such that:
      - g[sz-1, 0:sz] are all x
      - g[0:sz, sz-1] are all x
    Then the legend color set is the set of colors in the interior (0:sz-1, 0:sz-1),
    excluding bg and excluding the border color x.
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


def _majority_color(coords: np.ndarray, g: np.ndarray, banned: set[int]) -> int | None:
    """
    Return the unique intended majority among colors not in `banned`.

    Ties are broken by the first color that reaches the maximum while scanning
    component coordinates in row-major order. The generator avoids ties.
    """
    counts: dict[int, int] = {}
    best_color: int | None = None
    best_count = -1
    for r, c in coords:
        color = int(g[int(r), int(c)])
        if color in banned:
            continue
        counts[color] = counts.get(color, 0) + 1
        if counts[color] > best_count:
            best_color = color
            best_count = counts[color]
    return best_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove legend-colored cells from matching components, then homogenize what remains.

    Input structure:
    - A background color fills most of the board, and a small legend sits in the top-left corner.
    - The legend is bounded by one color along its bottom and right edges, while its interior shows the target color set.
    - Away from the legend are several 4-connected multicolor components placed on the background.

    Rule:
    - Read the target colors from the legend interior.
    - For each non-background connected component, test whether it contains all target colors.
    - If it does, replace target-colored pixels in that component with the background color, then recolor every surviving pixel in that same component to the most frequent non-target color found there.
    - Components missing at least one target color are left unchanged.

    Key insight: The legend still selects which components are affected, but the effect is now local editing inside each matching component rather than erasing the whole component.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if not legend_colors.issubset(cols):
            continue

        fill_color = _majority_color(pts, g, legend_colors)
        for r, c in pts:
            r = int(r)
            c = int(c)
            if int(g[r, c]) in legend_colors:
                out[r, c] = bg
            elif fill_color is not None:
                out[r, c] = int(fill_color)
    return out


# --------------- generator (D6 variant) ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)
MATCH_MIN_AREA = 12

# -- Kept-component fill density --
KEEP_DENSITY_LO = 0.10
KEEP_DENSITY_HI = 0.55

# -- Matching-component appearance --
MATCH_TARGET_MAX = 6
MATCH_MUTATE_MAX = 8

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Generation retry budget --
MAX_RETRIES = 300


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box (border color 3) at top-left and place target colors."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    # Force one target to stay isolated from the border-connected pair so the legend
    # itself is not mistaken for a matching component.
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


def _component_dims(matching: bool) -> tuple[int, int]:
    """Sample component dimensions, keeping matching components large enough to show holes clearly."""
    for _ in range(20):
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h, w = random.randint(*TALL_H), random.randint(*TALL_W)
        elif mode == "wide":
            h, w = random.randint(*WIDE_H), random.randint(*WIDE_W)
        else:
            h, w = random.randint(*MED_H), random.randint(*MED_W)
        if (not matching) or h * w >= MATCH_MIN_AREA:
            return h, w
    return max(3, MATCH_MIN_AREA // 3), 4


def _component_patch_keep(h: int, w: int, targets: list[int]) -> np.ndarray:
    """Build an unchanged component: rectangle of 0 with sparse colored pixels, missing at least one target."""
    area = h * w
    lo = max(1, math.ceil(area * KEEP_DENSITY_LO))
    hi = max(lo, math.floor(area * KEEP_DENSITY_HI))
    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    used_target_count = random.randint(0, NUM_TARGETS - 1)
    allowed = non_targets + random.sample(targets, used_target_count)
    colors = [random.choice(allowed) for _ in range(n_colored)]

    random.shuffle(colors)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _component_patch_matching(h: int, w: int, targets: list[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build a matching component and its transformed output.

    Output patch: one dominant non-target color everywhere except background holes
    where target-color pixels were deleted.
    Input patch: replace those holes with target colors, and mutate some surviving
    cells to 0 / other non-target colors while keeping the dominant color a clear majority.
    """
    area = h * w
    all_coords = [(r, c) for r in range(h) for c in range(w)]
    non_targets = [v for v in PALETTE if v not in targets]
    dominant = random.choice(non_targets)

    max_targets = min(MATCH_TARGET_MAX, area - 1)
    n_target_cells = random.randint(NUM_TARGETS, max_targets)
    target_coords = random.sample(all_coords, n_target_cells)
    target_coord_set = set(target_coords)

    out_patch = np.full((h, w), dominant, dtype=int)
    for r, c in target_coords:
        out_patch[r, c] = BG

    inp_patch = out_patch.copy()
    target_colors = list(targets) + [random.choice(targets) for _ in range(n_target_cells - NUM_TARGETS)]
    random.shuffle(target_colors)
    for (r, c), color in zip(target_coords, target_colors):
        inp_patch[r, c] = color

    residual_coords = [p for p in all_coords if p not in target_coord_set]
    residual_count = len(residual_coords)
    if residual_count >= 8:
        min_mutate = 2
    elif residual_count >= 5:
        min_mutate = 1
    else:
        min_mutate = 0
    max_mutate = min(MATCH_MUTATE_MAX, max(0, residual_count // 2 - 1))
    if max_mutate >= min_mutate:
        n_mutate = random.randint(min_mutate, max_mutate)
    else:
        n_mutate = 0

    if n_mutate > 0:
        mutate_coords = random.sample(residual_coords, n_mutate)
        mutate_palette = [0] + [c for c in non_targets if c != dominant]
        for r, c in mutate_coords:
            if 0 in mutate_palette and random.random() < 0.45:
                inp_patch[r, c] = 0
            else:
                inp_patch[r, c] = random.choice(mutate_palette)

    # Validate the intended effect locally.
    fill_color = _majority_color(np.argwhere(inp_patch != BG), inp_patch, set(targets))
    assert fill_color == dominant
    return inp_patch, out_patch, dominant


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid (input, output) pair; return input or None on failure."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for matching in roles:
        h, w = _component_dims(matching)
        if matching:
            in_patch, out_patch, _dominant = _component_patch_matching(h, w, targets)
        else:
            in_patch = _component_patch_keep(h, w, targets)
            out_patch = in_patch.copy()
        try:
            r, c = random_free_location(
                inp, np.zeros_like(in_patch), transparent=BG, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, in_patch, r, c)
        paste(out, out_patch, r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
