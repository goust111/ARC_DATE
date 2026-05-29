# Combo variant of d59b0160 (index 22)
# combo_spec: D12/2+D17/2
# Name: 忽略首次出现的颜色+最小匹配分量保留为种子
#
# Design analysis
# ===============
# fusion_rule:
#   - 先沿用原题的图例读取方式: 左上角 4x4 图例内部给出 3 个目标颜色, 外部对象仍是由 0 色矩形底板承载的
#     4 连通多色分量。
#   - 匹配判定先应用 D12/2: 对每个外部分量按行优先扫描, 每种颜色第一次出现不计入匹配集合, 只有第二次
#     出现时该颜色才算“有效出现”; 这与“该颜色在分量中至少出现 2 次”等价。
#   - 然后应用 D17/2: 只在这些“重复意义下匹配图例”的分量里比较面积, 面积最小者不删除, 而是整块改涂成
#     图例边框色 3 作为 seed; 其他匹配分量删除为背景。
#   - 因而完整逻辑不是“先按原题匹配再留最小”, 而是“先过重复计数门槛, 再在过门槛者里选最小 seed”。
#
# behavior_trace:
#   - 例: 图例颜色为 {1,4,8}。有三个外部分量 A/B/C。
#   - A 面积 9, 颜色多重集含 {1x2,4x2,8x2,...}; 原题会删, 融合规则里它也匹配, 且若最小则改成纯 3 色 seed。
#   - B 面积 18, 颜色多重集含 {1x2,4x2,8x2,...}; 原题会删, 融合规则里也匹配, 但因不是最小而被删除。
#   - C 面积 16, 肉眼能看到 1/4/8 都出现, 但其中 8 只出现 1 次; 原题会删, 融合规则里 8 不进入匹配集合,
#     所以 C 保留原样。这让 D12 与 D17 在同一组对象上同时可见。
#
# complexity_calibration:
#   - 若 kept 分量只是简单缺少某个图例色, 则样例仍可能看起来像原题; 因此生成器强制至少一个 near_miss 分量:
#     它可见地包含全部图例色, 但恰有一个目标色只出现 1 次。
#   - 若只有一个重复匹配分量, 则看不出“删大留小”; 因此生成器强制至少 2 个重复匹配分量, 且最小面积唯一。
#   - 颜色变化只使用已有边框色 3 作为 seed 标记, 不额外引入新映射, 使输出规律足够集中且可从少量样例推断。
#
# implicit_assumption_review:
#   - 原 solver 用 `components_mask(g != bg, connectivity=4)` 找分量, 这个前提仍成立, 因为 0 色底板必须继续参与连通。
#   - 原 solver 用颜色集合 `{g[r,c]}` 判匹配, 这个前提失效: 单次出现的图例色不应算数, 所以要改成重复颜色提取器。
#   - 原 solver 对每个匹配分量立刻删除, 这个前提也失效: 现在需要先收集全部匹配者, 再决定谁是 seed、谁被删除。
#   - 图例本身虽然通常不会通过重复匹配, 但“最小匹配分量比较”阶段不该依赖该偶然事实, 所以 solver 显式跳过完全位于
#     图例方块内的分量。
#
# placement_legality:
#   - 所有对象仍作为完整矩形 patch 放置, 用 `random_free_location(..., padding=1)` 保证彼此及图例之间至少隔 1 格背景。
#   - seed 候选使用小矩形, 其他匹配分量强制面积更大, 从构造上保证唯一最小匹配分量。
#   - deleted/seed 分量至少含 6 个彩点(3 个目标色各 2 次); near_miss 至少含 5 个目标彩点(2,2,1), 均在尺寸范围内可实现。
#
# visual_observability_and_cleanliness:
#   - 人类可直接数出“某目标色是否重复出现”, 不依赖颜色编号或扫描顺序这类不可见因素。
#   - 输出里的纯 3 色 seed 与被擦除的其他匹配分量形成对照, 同时 near_miss 继续保留原貌, 三类命运清晰可分。
#   - 同一区域不叠加多重标记: 输入只有图例和矩形对象, 输出只多了一个纯色 seed, 画面不会杂乱。
#
# generator_guarantees_for_solver_assumptions:
#   - 至少 2 个外部分量在“重复计数”规则下匹配图例。
#   - 最小匹配面积唯一, 因而 solver 不需要依赖不可见 tie-break 才能与样例一致。
#   - 至少 1 个 kept near_miss 分量包含全部图例色但失败于重复计数, 确保融合行为与原题输出不同。

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the outer border."""
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


def _repeated_match_colors(g: np.ndarray, pts: np.ndarray) -> set[int]:
    """Return the colors that appear at least twice inside the component."""
    seen_once: set[int] = set()
    repeated: set[int] = set()
    for r, c in sorted((int(r), int(c)) for r, c in pts):
        color = int(g[r, c])
        if color in seen_once:
            repeated.add(color)
        else:
            seen_once.add(color)
    return repeated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the smallest repeated-match component as a seed, and erase the other repeated matches.

    Input structure:
    - One background color fills most of the board.
    - The top-left corner contains a small legend square whose bottom and right edges share one border color.
    - Away from that legend sit several 4-connected multicolor rectangles built on color-0 bodies with scattered colored cells.

    Rule:
    - Read the target color set from the legend interior.
    - For each non-legend foreground component, scan its pixels in row-major order and ignore the first occurrence of every color.
    - A component counts as matching only if all legend colors survive that filter, meaning each target color appears at least twice in the component.
    - Among those matching components, recolor the unique smallest one to the legend-border color as a seed, erase every other match to background, and leave all non-matches unchanged.

    Key insight: The legend still names the required palette, but qualification now depends on repeated target colors, and size then decides which qualified component is specially protected.
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
        match_colors = _repeated_match_colors(g, pts)
        if legend_colors.issubset(match_colors):
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
DENSITY_LO = 0.12
DENSITY_HI = 0.55
MATCH_COUNTS = [2, 3]
MATCH_WEIGHTS = [5, 2]
NUM_KEPT = 3
MAX_RETRIES = 500

SMALL_MATCH_DIMS = [(3, 3), (3, 4), (4, 3), (4, 4)]
KEEP_SHAPE_WEIGHTS = [5, 3, 2]
MATCH_SHAPE_WEIGHTS = [5, 2, 3]
TALL_H = (6, 9)
TALL_W = (2, 4)
WIDE_H = (3, 4)
WIDE_W = (6, 8)
MED_H = (4, 6)
MED_W = (4, 6)
KEEP_H = (3, 6)
KEEP_W = (2, 6)


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the unchanged top-left legend in both input and output."""
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
    """Pick a compact seed rectangle so the smallest repeated-match area is visually obvious."""
    return random.choice(SMALL_MATCH_DIMS)


def _larger_match_dims(seed_area: int) -> tuple[int, int]:
    """Pick a matching rectangle strictly larger than the seed area."""
    for _ in range(100):
        h, w = _sample_mode_dims(MATCH_SHAPE_WEIGHTS)
        if h * w > seed_area:
            return h, w
    raise RuntimeError("failed to sample larger matching component")


def _keep_dims() -> tuple[int, int]:
    """Pick a moderate kept rectangle so the board remains packable."""
    if random.random() < 0.7:
        return random.randint(*KEEP_H), random.randint(*KEEP_W)
    return _sample_mode_dims(KEEP_SHAPE_WEIGHTS)


def _sample_colored_count(h: int, w: int, minimum: int) -> int:
    area = h * w
    lo = max(minimum, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    hi = min(hi, area)
    lo = min(lo, hi)
    return random.randint(lo, hi)


def _deleted_colors(targets: list[int], non_targets: list[int], n_colored: int) -> list[int]:
    colors = [color for color in targets for _ in range(2)]
    extras = n_colored - len(colors)
    colors.extend(random.choice(targets + non_targets) for _ in range(extras))
    random.shuffle(colors)
    return colors


def _near_miss_colors(targets: list[int], non_targets: list[int], n_colored: int) -> list[int]:
    weak = random.choice(targets)
    strong = [color for color in targets if color != weak]
    colors = [weak] + [strong[0], strong[0], strong[1], strong[1]]
    extras = n_colored - len(colors)
    colors.extend(random.choice(non_targets) for _ in range(extras))
    random.shuffle(colors)
    return colors


def _missing_colors(targets: list[int], non_targets: list[int], n_colored: int) -> list[int]:
    used_target_count = random.randint(0, NUM_TARGETS - 1)
    used_targets = random.sample(targets, used_target_count)
    colors: list[int] = []
    for color in used_targets:
        colors.extend([color] * random.randint(1, 2))
    while len(colors) > n_colored:
        colors.pop()
    extras = n_colored - len(colors)
    colors.extend(random.choice(non_targets) for _ in range(extras))
    random.shuffle(colors)
    return colors


def _component_patch(h: int, w: int, targets: list[int], style: str) -> np.ndarray:
    """Build a solid rectangle whose colored cells realize the requested repeated-color behavior."""
    non_targets = [v for v in PALETTE if v not in targets]
    if style in {"seed", "delete"}:
        n_colored = _sample_colored_count(h, w, minimum=2 * NUM_TARGETS)
        colors = _deleted_colors(targets, non_targets, n_colored)
    elif style == "near_miss":
        n_colored = _sample_colored_count(h, w, minimum=5)
        colors = _near_miss_colors(targets, non_targets, n_colored)
    elif style == "missing":
        n_colored = _sample_colored_count(h, w, minimum=1)
        colors = _missing_colors(targets, non_targets, n_colored)
    else:
        raise ValueError(f"unknown component style: {style}")

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _component_specs(targets: list[int]) -> list[dict]:
    """Describe one seed, 1-2 larger repeated matches, and 3 kept components."""
    seed_h, seed_w = _seed_dims()
    seed_area = seed_h * seed_w
    specs = [{"role": "seed", "patch": _component_patch(seed_h, seed_w, targets, "seed"), "area": seed_area}]

    match_count = random.choices(MATCH_COUNTS, weights=MATCH_WEIGHTS)[0]
    for _ in range(match_count - 1):
        h, w = _larger_match_dims(seed_area)
        specs.append({"role": "delete", "patch": _component_patch(h, w, targets, "delete"), "area": h * w})

    near_miss_count = random.randint(1, 2)
    for _ in range(near_miss_count):
        h, w = _keep_dims()
        specs.append({"role": "near_miss", "patch": _component_patch(h, w, targets, "near_miss"), "area": h * w})
    for _ in range(NUM_KEPT - near_miss_count):
        h, w = _keep_dims()
        specs.append({"role": "missing", "patch": _component_patch(h, w, targets, "missing"), "area": h * w})
    return specs


def _place_components(inp: np.ndarray, out: np.ndarray, specs: list[dict]) -> bool:
    """Place all rectangles with 1-cell separation and construct the intended output."""
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
        role = spec["role"]
        if role in {"near_miss", "missing"}:
            paste(out, patch, r, c)
        elif role == "seed":
            paste(out, np.full_like(patch, BORDER), r, c)
    return True


def _matching_component_areas(grid: np.ndarray) -> list[int]:
    """Areas of external components that match under the repeated-color rule."""
    bg = _bg_boundary(grid)
    legend_size, _border, legend_colors = _find_legend(grid, bg)
    areas = []
    for pts in components_mask(grid != bg, connectivity=4):
        if np.all(pts[:, 0] < legend_size) and np.all(pts[:, 1] < legend_size):
            continue
        if legend_colors.issubset(_repeated_match_colors(grid, pts)):
            areas.append(int(len(pts)))
    return sorted(areas)


def _has_near_miss_against_original(grid: np.ndarray) -> bool:
    """
    Check that at least one external component would match the original task
    but fails the repeated-color rule.
    """
    bg = _bg_boundary(grid)
    legend_size, _border, legend_colors = _find_legend(grid, bg)
    for pts in components_mask(grid != bg, connectivity=4):
        if np.all(pts[:, 0] < legend_size) and np.all(pts[:, 1] < legend_size):
            continue
        colors = {int(grid[r, c]) for r, c in pts}
        repeated = _repeated_match_colors(grid, pts)
        if legend_colors.issubset(colors) and not legend_colors.issubset(repeated):
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt with one protected smallest repeated-match component."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)
    specs = _component_specs(targets)

    if not _place_components(inp, out, specs):
        return None

    match_areas = _matching_component_areas(inp)
    if len(match_areas) < 2 or match_areas[0] == match_areas[1]:
        return None
    if not _has_near_miss_against_original(inp):
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
