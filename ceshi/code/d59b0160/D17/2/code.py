# Variant of d59b0160
# Dimension: D17 — 分量面积对匹配的影响 (meta)
# Change: 不考虑分量面积，任何大小的匹配分量都被删除... → 面积最小的匹配分量被保留为'种子'

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the D17 variant
# -----------------------------------
# spec:
#   preserved_core:
#     - 左上角 4x4 图例仍然给出 3 个目标颜色。
#     - 网格其余对象仍然是放在背景上的 4 连通多色矩形分量。
#     - "匹配分量" 仍然定义为: 颜色集合包含全部 3 个图例颜色。
#   changed_behavior:
#     - 原题: 所有匹配分量都删除。
#     - 变体: 外部分量中, 面积最小的匹配分量保留, 并整体改涂为种子色 3;
#       其他匹配分量照旧删除。
#   concrete_pixel_effect:
#     - 输入里会出现至少 2 个都含有全部图例颜色的矩形分量。
#     - 输出里, 其中最小的那个不会消失, 而是变成纯绿色矩形;
#       其余匹配矩形被擦成背景。
#     - 不匹配分量保持原状。
#   behavior_trace:
#     - 例: 图例颜色为 {1,4,8}; 有三个外部分量 A/B/C.
#     - A 面积 12, 颜色含 {0,1,4,8}; B 面积 20, 也含 {1,4,8}; C 缺少 8.
#     - 原题输出: A 和 B 都被删除, C 保留。
#     - 本变体输出: A 改成纯 3 色种子, B 删除, C 保留。
#     - 差异集中在 "匹配分量命运判定" 这一步, 输入识别逻辑不变。
#   complexity_calibration:
#     - 只在匹配分量内部引入面积比较, 不引入额外排序维度或颜色映射。
#     - 生成器强制至少 2 个匹配分量, 避免退化成 "只有一个匹配分量, 看不出删大留小"。
#     - 生成器强制最小匹配面积唯一, 避免样例依赖隐藏 tie-break。
#   implicit_assumption_review:
#     - 原 solver 按 (g != bg) 找全部分量, 默认图例自身不会误判为待删对象。
#     - 变体里若仍把图例碎片纳入最小面积比较, 可能把图例误当种子;
#       因此 solver 显式跳过完全位于图例方块内的分量。
#     - 原 solver 对每个匹配分量都直接清空; 该写法依赖 "所有匹配分量命运相同"。
#       变体中此假设失效, 必须先收集全部匹配分量, 再决定哪个保留、哪些删除。
#   placement_legality:
#     - 每个对象仍作为完整矩形 patch 放置, 用 1 格 padding 保证互不接触。
#     - seed/matched/kept 全都在图例外部随机布局, 不越界不重叠。
#     - 匹配分量的面积由其矩形 bbox 决定, 与内部彩点数量无关, 因而 solver/generator 一致。
#   visual_observability:
#     - 绿色 3 本来只出现在图例边框; 输出把 seed 整体改成 3, 肉眼能马上看到"特殊保留者"。
#     - 删除对象与绿色 seed 会在同一类 "全含图例色" 对象中同时出现, 规则可从 2-3 例中归纳。
#   cleanliness:
#     - 同一网格只叠加三类元素: 图例, 普通矩形对象, 输出中的纯绿色 seed。
#     - 不再加入额外标记符号, 避免多层覆盖造成视觉混乱。


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color on the outer frame."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Find the top-left legend square whose bottom row and right column share one border color.
    Return (size, border_color, interior_target_colors).
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        border_color = int(g[sz - 1, sz - 1])
        if border_color == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == border_color) and np.all(g[0:sz, sz - 1] == border_color):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(border_color)
            return int(sz), int(border_color), colors
    raise AssertionError("no legend border found in grid")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the smallest matching component as a seed and erase the larger matches.

    Input structure:
    - One background color occupies most of the board.
    - The top-left corner contains a small legend square whose interior shows the target palette and whose bottom/right edges share one border color.
    - Away from the legend are several 4-connected multicolor components built on the background.

    Rule:
    - Read the target color set from the legend interior.
    - Among non-legend connected components, collect those whose color set contains every target color.
    - Compare only those matching components by area: recolor the smallest one to the legend-border color as a seed, and erase every other matching component to background.
    - Leave all non-matching components unchanged.

    Key insight: The legend still defines which components qualify, but size now breaks the symmetry inside that qualifying set: the smallest full-palette component survives as a highlighted seed.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_size, border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    matches: list[tuple[int, int, int, np.ndarray]] = []
    for pts in components_mask(g != bg, connectivity=4):
        if np.all(pts[:, 0] < legend_size) and np.all(pts[:, 1] < legend_size):
            continue
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            area = int(len(pts))
            top = int(pts[:, 0].min())
            left = int(pts[:, 1].min())
            matches.append((area, top, left, pts))

    if not matches:
        return out

    seed_idx = min(range(len(matches)), key=lambda i: matches[i][:3])
    for idx, (_area, _top, _left, pts) in enumerate(matches):
        fill = border_color if idx == seed_idx else bg
        for r, c in pts:
            out[r, c] = fill
    return out


# --------------- generator (o5x) ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component synthesis --
DENSITY_LO = 0.10
DENSITY_HI = 0.55
MATCH_COUNTS = [2, 3]
MATCH_WEIGHTS = [5, 2]
NUM_KEPT = 3
MAX_RETRIES = 400

SMALL_MATCH_DIMS = [(3, 3), (3, 4), (4, 3), (4, 4)]
KEEP_SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
MATCH_SHAPE_WEIGHTS = [5, 2, 3]  # medium, tall, wide
TALL_H = (6, 9)
TALL_W = (2, 4)
WIDE_H = (3, 4)
WIDE_W = (6, 8)
MED_H = (4, 6)
MED_W = (4, 6)
KEEP_H = (3, 6)
KEEP_W = (2, 6)


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the top-left legend; the output legend is unchanged."""
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


def _sample_mode_dims(weights: list[int]) -> tuple[int, int]:
    """Sample rectangle dimensions from medium / tall / wide modes."""
    mode = random.choices(["medium", "tall", "wide"], weights=weights)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _seed_dims() -> tuple[int, int]:
    """Pick a small seed rectangle so the smallest match is visually obvious."""
    return random.choice(SMALL_MATCH_DIMS)


def _larger_match_dims(min_area: int) -> tuple[int, int]:
    """Pick a matching component strictly larger than the seed area."""
    for _ in range(100):
        h, w = _sample_mode_dims(MATCH_SHAPE_WEIGHTS)
        if h * w > min_area:
            return h, w
    raise RuntimeError("failed to sample larger matching component")


def _keep_dims() -> tuple[int, int]:
    """Pick a kept component; sizes remain moderate to keep packing feasible."""
    if random.random() < 0.7:
        return random.randint(*KEEP_H), random.randint(*KEEP_W)
    return _sample_mode_dims(KEEP_SHAPE_WEIGHTS)


def _component_patch(h: int, w: int, targets: list[int], match: bool) -> np.ndarray:
    """Build a solid 4-connected rectangle with sparse colored pixels over color 0."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if match:
        lo = max(lo, NUM_TARGETS)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if match:
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


def _component_spec(role: str, targets: list[int], seed_area: int | None = None) -> dict:
    """Describe one component before placement."""
    if role == "seed":
        h, w = _seed_dims()
        patch = _component_patch(h, w, targets, match=True)
    elif role == "delete":
        assert seed_area is not None
        h, w = _larger_match_dims(seed_area)
        patch = _component_patch(h, w, targets, match=True)
    else:
        h, w = _keep_dims()
        patch = _component_patch(h, w, targets, match=False)
    return {"role": role, "patch": patch, "area": h * w}


def _place_components(inp: np.ndarray, out: np.ndarray, specs: list[dict]) -> bool:
    """Place all components with 1-cell separation and build the expected output."""
    for spec in sorted(specs, key=lambda x: x["area"], reverse=True):
        patch = spec["patch"]
        try:
            r, c = random_free_location(
                inp,
                np.zeros_like(patch),
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return False
        paste(inp, patch, r, c)
        if spec["role"] == "keep":
            paste(out, patch, r, c)
        elif spec["role"] == "seed":
            paste(out, np.full_like(patch, BORDER), r, c)
    return True


def _count_matching_components(grid: np.ndarray) -> list[int]:
    """Return areas of all external matching components in the input."""
    bg = _bg_boundary(grid)
    legend_size, _border, legend_colors = _find_legend(grid, bg)
    areas = []
    for pts in components_mask(grid != bg, connectivity=4):
        if np.all(pts[:, 0] < legend_size) and np.all(pts[:, 1] < legend_size):
            continue
        cols = {int(grid[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            areas.append(int(len(pts)))
    return sorted(areas)


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt with a guaranteed unique smallest matching component."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    roles = ["seed"] + ["delete"] * (random.choices(MATCH_COUNTS, weights=MATCH_WEIGHTS)[0] - 1) + ["keep"] * NUM_KEPT
    seed_h, seed_w = _seed_dims()
    seed_area = seed_h * seed_w
    specs = [{"role": "seed", "patch": _component_patch(seed_h, seed_w, targets, match=True), "area": seed_area}]
    for role in roles[1:]:
        specs.append(_component_spec(role, targets, seed_area=seed_area))

    if not _place_components(inp, out, specs):
        return None

    match_areas = _count_matching_components(inp)
    if len(match_areas) < 2 or match_areas[0] == match_areas[1]:
        return None
    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(out, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
