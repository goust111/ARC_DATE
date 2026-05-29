# Variant of d59b0160
# Dimension: D16 — 分量中颜色的空间分布约束 (meta)
# Change: 仅检查分量是否包含图例颜色，不关注颜色在分量内的空间分布
#   → 图例颜色必须在分量中空间相邻
#
# Design analysis:
# spec_variant:
#   concretization:
#     - 仍然保留左上角 4x4 图例、右边和下边为颜色 3 的边框、背景为颜色 7。
#     - 网格中的主体对象仍是彼此分离的实心矩形分量，矩形底色为 0，内部散布若干彩色像素。
#     - 与原题不同，删除条件不再是“分量里出现了全部图例色”。
#       现在必须在同一分量内部看到这些图例色彼此贴边接触；对 3 个目标色来说，
#       需要每一对目标色之间都存在至少一对正交相邻像素。
#     - 因此 generator 会显式制造两种都“含全目标色”的分量：
#       1) 删除分量：包含一个 2x2 的见证块，其中一个目标色重复一次，从而三种目标色两两都接触。
#       2) 保留分量：三种目标色各出现一次，但没有任何重复目标色，因此不可能形成完整的两两接触图。
#   behavior_trace:
#     - 原逻辑：只要一个分量的颜色集合 superset(图例色集合) 就删除。
#     - 新逻辑：先要求颜色集合包含全部图例色，再检查目标色之间的接触边是否构成完全图；
#       若 A/B/C 只是被 0 底色隔开地散落在同一大矩形中，则保留。
#     - 具体例子：若某个矩形里有 A、B、C 三色各一个且互不重复，原题会删；本变体会保留。
#   complexity_calibration:
#     - 目标色数量仍固定为 3，避免把“完整接触图”复杂度推得太高。
#     - 每个样例中固定保留至少一个 `kept_full` 分量，让 D16 与原题的差异在少量样例中可见。
#   implicit_assumption_audit:
#     - 原 `main()` 中 `legend_colors.issubset(cols)` 默认“只要出现过即可”，该前提已失效，必须换成显式邻接图检查。
#     - 原 generator 对 deleted 分量只保证“放入全部目标色”即可，这在新规则下不够；现在必须构造见证块。
#     - 原 generator 的 kept 分量允许随机混入部分目标色；在新规则下若任意追加目标色会意外制造接触，所以保留分量的额外彩色像素限制为非目标色。
#   placement_legality:
#     - 仍用 `random_free_location(..., padding=1, padding_connectivity=8)` 放置整块矩形，保证分量之间不重叠也不相碰。
#     - 所有局部图案都在矩形内部构造；删除见证块只需要 2x2，符合原尺寸范围。
#   visual_observability:
#     - 人类可见的关键信号是：有些被保留的大矩形明明含有图例中的三种颜色，但三色没有挤成相互贴边的小团块。
#     - 与之对应，被删除的矩形中会出现一个更紧凑的目标色接触团。
#   neatness:
#     - 仍然使用分离矩形，不叠加额外标记，不引入新的区域类型，视觉结构与原题一致。
#   color_inference:
#     - 不引入任何依赖颜色 id 的排序或变换；颜色只作为“是否属于图例目标色”与“是否接触”的可见属性。

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


def _legend_colors_pairwise_touch(g: np.ndarray, pts: np.ndarray, legend_colors: set[int]) -> bool:
    """
    Return True iff all legend colors are present in this component and every pair of
    legend colors has at least one 4-neighbor contact edge inside the component.
    """
    present = {int(g[r, c]) for r, c in pts if int(g[r, c]) in legend_colors}
    if present != legend_colors:
        return False

    pt_set = {tuple(map(int, rc)) for rc in pts}
    touches = {color: set() for color in legend_colors}

    for r, c in pt_set:
        color = int(g[r, c])
        if color not in legend_colors:
            continue
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if (nr, nc) not in pt_set:
                continue
            other = int(g[nr, nc])
            if other in legend_colors and other != color:
                touches[color].add(other)
                touches[other].add(color)

    return all((legend_colors - {color}).issubset(touches[color]) for color in legend_colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove only those connected components whose legend colors physically touch each other.

    Input structure:
    - The background color is read from the outer frame.
    - The top-left corner contains a small legend square; its bottom and right edges share one border color, while the interior lists the target colors.
    - Elsewhere, the non-background cells form separate multicolor 4-connected components.

    Rule:
    - Read the target color set from inside the legend.
    - Examine each 4-connected component independently.
    - Delete a component only if it contains every target color and every pair of target colors has a direct 4-neighbor contact somewhere inside that component.
    - If the target colors are merely present but separated by zero-cells or other colors, keep the component unchanged.

    Key insight: The legend now specifies not just a palette but a local topology: the target colors must meet one another inside the same object, not simply coexist in it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        if _legend_colors_pairwise_touch(g, pts, legend_colors):
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (D16 variant) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

DENSITY_LO = 0.10
DENSITY_HI = 0.55

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

MAX_RETRIES = 300


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the legend while keeping its three target colors split across legend subcomponents."""
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
    """Sample component (h, w) according to shape-mode weights."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _color_budget(h: int, w: int, min_count: int) -> int:
    area = h * w
    lo = max(min_count, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    return random.randint(lo, hi)


def _all_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w)]


def _component_patch(h: int, w: int, targets: list[int], role: str) -> np.ndarray:
    """
    Build an h x w solid-0 component.

    Roles:
    - deleted: contains a 2x2 witness block whose target-color contact graph is complete.
    - kept_full: contains all target colors exactly once, so they cannot realize all pairwise contacts.
    - kept_partial: omits at least one target color.
    """
    patch = np.zeros((h, w), dtype=int)
    non_targets = [v for v in PALETTE if v not in targets]
    free = set(_all_coords(h, w))

    if role == "deleted":
        n_colored = _color_budget(h, w, 4)
        top = random.randint(0, h - 2)
        left = random.randint(0, w - 2)
        trio = targets[:]
        random.shuffle(trio)
        motif = [
            ((top, left), trio[0]),
            ((top, left + 1), trio[1]),
            ((top + 1, left), trio[2]),
            ((top + 1, left + 1), trio[1]),
        ]
        for coord, color in motif:
            patch[coord] = color
            free.discard(coord)
    elif role == "kept_full":
        n_colored = _color_budget(h, w, NUM_TARGETS)
        coords = random.sample(list(free), NUM_TARGETS)
        for color, coord in zip(targets, coords):
            patch[coord] = color
            free.discard(coord)
    elif role == "kept_partial":
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        n_colored = _color_budget(h, w, used_target_count)
        coords = random.sample(list(free), used_target_count)
        for color, coord in zip(random.sample(targets, used_target_count), coords):
            patch[coord] = color
            free.discard(coord)
    else:
        raise ValueError(f"unknown role: {role}")

    remaining = n_colored - int(np.count_nonzero(patch))
    if remaining > 0:
        filler_coords = random.sample(list(free), remaining)
        filler_colors = [random.choice(non_targets) for _ in range(remaining)]
        for coord, color in zip(filler_coords, filler_colors):
            patch[coord] = color

    return patch


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose output is recovered by `main`."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - NUM_KEPT
    kept_full_count = random.randint(1, 2)
    kept_partial_count = NUM_KEPT - kept_full_count
    roles = (
        ["deleted"] * deleted_count
        + ["kept_full"] * kept_full_count
        + ["kept_partial"] * kept_partial_count
    )
    random.shuffle(roles)

    for role in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, role)
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
        if role != "deleted":
            paste(out, patch, r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
