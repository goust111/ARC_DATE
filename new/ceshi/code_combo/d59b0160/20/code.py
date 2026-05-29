# Combo variant of d59b0160 (index 20)
# combo_spec: D1/4+D6/2
# Name: 多数颜色投票删除+删除后剩余重着色
#
# Design analysis
# design_analysis:
#   concrete_behavior:
#   - 输入仍是 16x16 背景网格，左上角 4x4 图例用绿色边框给出 3 个目标色，其他位置放若干与背景 7 分离的 4-连通矩形组件。
#   - 原题的触发条件是“组件必须同时含有图例中的全部颜色”；融合后改成“组件内属于图例色的像素数超过该组件总像素的一半”。
#   - 命中后也不再整块清空：先只删除图例色像素，再把残留的非图例色像素统一刷成该残留部分中出现最多的那种颜色。
#   - 例：图例色为 {1,4,9}，某组件共有 12 格，其中 7 格是 1/4、其余 5 格是 {0,6,6,8,6}。原题因为缺少 9 会保留整块；本融合变体因为图例色占 7/12>1/2，会把那 7 格删成背景 7，再把剩下的 0/6/6/8/6 全部改成 6。
#   - 对比另一类组件：若它含有 {1,4,9} 但仅 4/12 格属于图例色，则原题会删除，而本融合变体会完全保留。这两种对照需要同时出现在样本里，才能让“多数投票”可观察。
#   complexity_calibration:
#   - 生成器固定制造两类关键对象：至少一个“图例色占多数但缺少某个图例颜色”的命中组件，至少一个“包含全部图例色但图例色不占多数”的保留组件。
#   - 这样能同时排除退化为原题（ALL 匹配）和退化为仅 D6/2（仍需 ALL 匹配）的可能；解题者会看到有些缺色组件被改写，也会看到有些全集组件没被改写。
#   - 命中组件的残留非图例色会围绕单一显著多数色构造，并混入少量其他非图例色，保证“删除后重着色”在输出中肉眼可见，而不是输入本来就近乎单色。
#   implicit_assumptions_review:
#   - `_bg_boundary` 仍假设外框多数是背景色 7；生成器保持这一点。
#   - `_find_legend` 仍假设左上图例由同色下边和右边围成；图例结构保持不变。
#   - 原版 `main()` 可以把所有非背景连通分量一视同仁处理，因为 legend 本身不会满足“包含全部图例色”。改成“图例色占多数”后，legend 内部的孤立目标像素会错误命中，因此 solver 必须显式跳过完全落在 legend 方框内的分量。
#   - 新版重着色步骤假设命中组件删掉图例色后仍有非图例残留；solver 对空残留做兼容处理，但生成器始终保证命中组件至少保留 4 格非图例像素。
#   - 多数判定依赖组件总面积，因此组件必须在输入中保持单个 4-连通整体；生成器继续用整块矩形 patch 作为组件载体。
#   placement_validity:
#   - 所有非图例组件通过 `random_free_location(..., padding=1, padding_connectivity=8)` 放置，保证彼此与 legend 至少隔 1 格背景，不重叠、不越界。
#   - 匹配组件的输出会出现背景孔洞，但这只发生在输出，不会破坏输入阶段的连通分量识别。
#   visual_observability:
#   - 命中组件在输出中会保留轮廓、出现若干与图例色对应的空洞，并被统一刷成一种非图例颜色；未命中组件则保持原样，视觉对比明确。
#   - 由于样本里同时存在“缺少某个图例色却被改写”的组件和“包含全部图例色却未改写”的组件，人类可以从 2-4 个样例中归纳出规则是面积占比而不是颜色种类全集。
#   cleanliness_and_color_inference:
#   - 图例始终局限在左上 4x4，求解时完全跳过该区域内分量，空间角色清晰。
#   - 每个命中组件只对应一个明显的残留多数色，避免并列多数造成不可见的 tie-break 规则；生成器显式避免并列。

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the top-left legend whose bottom edge and right edge share one border color.

    Returns:
      (legend_size, border_color, interior_target_colors)
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


def _majority_non_legend_color(coords: np.ndarray, g: np.ndarray, legend_colors: set[int]) -> int | None:
    """Return the most frequent color among component pixels not in the legend palette."""
    counts: dict[int, int] = {}
    best_color: int | None = None
    best_count = -1
    for r, c in coords:
        color = int(g[int(r), int(c)])
        if color in legend_colors:
            continue
        counts[color] = counts.get(color, 0) + 1
        if counts[color] > best_count:
            best_color = color
            best_count = counts[color]
    return best_color


def _target_pixel_count(coords: np.ndarray, g: np.ndarray, legend_colors: set[int]) -> int:
    """Count component pixels whose color belongs to the legend palette."""
    total = 0
    for r, c in coords:
        if int(g[int(r), int(c)]) in legend_colors:
            total += 1
    return int(total)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Edit components whose legend-colored area forms a strict majority.

    Input structure:
    - Most cells are the background color, while a small framed legend sits in the top-left corner.
    - The legend uses one border color along its bottom and right edges; its interior pixels list the target palette.
    - Elsewhere are several 4-connected non-background components separated by background gaps.

    Rule:
    - Read the target color set from inside the legend.
    - For each non-background component outside the legend box, count how many of its pixels use legend colors.
    - If those legend-colored pixels make up more than half of the component, delete only those pixels to background, then recolor every remaining pixel in that component to the most frequent non-legend color from the original component.
    - Components whose legend-colored area is at most half are left unchanged.

    Key insight: The legend now defines a voting palette rather than a required full set: components are selected by target-color area ratio, then locally hollowed out and homogenized by their surviving non-legend majority color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        if np.all(pts[:, 0] < sz) and np.all(pts[:, 1] < sz):
            continue

        total = int(pts.shape[0])
        target_count = _target_pixel_count(pts, g, legend_colors)
        if target_count * 2 <= total:
            continue

        fill_color = _majority_non_legend_color(pts, g, legend_colors)
        for r, c in pts:
            r = int(r)
            c = int(c)
            if int(g[r, c]) in legend_colors:
                out[r, c] = bg
            elif fill_color is not None:
                out[r, c] = int(fill_color)
    return out


# --------------- generator (combo variant) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

SHAPE_WEIGHTS = [5, 2, 2]  # medium, tall, wide
TALL_H = (7, 11)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 8)
MED_H = (3, 6)
MED_W = (3, 6)

MATCH_MIN_AREA = 10
RESIDUAL_MIN = 4
MATCH_MUTATE_MAX = 5
KEEP_MUTATE_MAX = 6

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [2, 5, 3]
NUM_KEPT = 3

MAX_RETRIES = 400


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the top-left legend while keeping one target isolated from border-connected legend cells."""
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


def _component_dims(matching: bool) -> tuple[int, int]:
    """Sample component dimensions, keeping matched components large enough for holes plus residual mass."""
    for _ in range(30):
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h, w = random.randint(*TALL_H), random.randint(*TALL_W)
        elif mode == "wide":
            h, w = random.randint(*WIDE_H), random.randint(*WIDE_W)
        else:
            h, w = random.randint(*MED_H), random.randint(*MED_W)
        if (not matching) or h * w >= MATCH_MIN_AREA:
            return int(h), int(w)
    return 5, 4


def _assign_target_colors(
    n_cells: int,
    targets: list[int],
    *,
    require_all: bool,
) -> list[int]:
    """Color a set of target cells using either all target colors or only a strict subset."""
    if require_all:
        used = list(targets)
    else:
        used_count = random.randint(1, NUM_TARGETS - 1)
        used = random.sample(targets, used_count)
    colors = list(used)
    while len(colors) < n_cells:
        colors.append(random.choice(used))
    random.shuffle(colors)
    return colors


def _pick_dominant_color(targets: list[int]) -> int:
    """Choose a non-legend dominant color for a component."""
    non_targets = [0] + [v for v in PALETTE if v not in targets]
    return int(random.choice(non_targets))


def _other_non_legend_colors(targets: list[int], dominant: int) -> list[int]:
    """List alternative non-legend colors that can appear without becoming the residual majority."""
    return [c for c in ([0] + PALETTE) if c not in set(targets) | {dominant, BG, BORDER}]


def _component_patch_matching(
    h: int,
    w: int,
    targets: list[int],
    *,
    require_all_targets: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a matched component.

    The input has a strict majority of legend colors. The output deletes those
    legend-colored cells to background, then recolors the residual to the unique
    dominant non-legend color.
    """
    area = h * w
    max_target_cells = area - RESIDUAL_MIN
    min_target_cells = max(NUM_TARGETS if require_all_targets else 1, area // 2 + 1)
    if max_target_cells < min_target_cells:
        raise ValueError("matching component too small for required majority-with-residual layout")

    dominant = _pick_dominant_color(targets)
    all_coords = [(r, c) for r in range(h) for c in range(w)]
    target_count = random.randint(min_target_cells, max_target_cells)
    target_coords = random.sample(all_coords, target_count)
    target_coord_set = set(target_coords)

    out_patch = np.full((h, w), dominant, dtype=int)
    for r, c in target_coords:
        out_patch[r, c] = BG

    inp_patch = out_patch.copy()
    target_colors = _assign_target_colors(target_count, targets, require_all=require_all_targets)
    for (r, c), color in zip(target_coords, target_colors):
        inp_patch[r, c] = color

    residual_coords = [p for p in all_coords if p not in target_coord_set]
    alt_colors = _other_non_legend_colors(targets, dominant)
    max_mutate = min(MATCH_MUTATE_MAX, max(0, len(residual_coords) // 3))
    if alt_colors and max_mutate > 0:
        n_mutate = random.randint(1, max_mutate)
        for r, c in random.sample(residual_coords, n_mutate):
            inp_patch[r, c] = random.choice(alt_colors)

    fill_color = _majority_non_legend_color(np.argwhere(inp_patch != BG), inp_patch, set(targets))
    assert fill_color == dominant
    assert _target_pixel_count(np.argwhere(inp_patch != BG), inp_patch, set(targets)) * 2 > area
    if require_all_targets:
        assert set(targets).issubset({int(inp_patch[r, c]) for r, c in target_coords})
    else:
        assert not set(targets).issubset({int(inp_patch[r, c]) for r, c in target_coords})
    return inp_patch, out_patch


def _component_patch_keep(
    h: int,
    w: int,
    targets: list[int],
    *,
    require_all_targets: bool,
) -> np.ndarray:
    """Build a kept component whose legend-colored share stays at or below half."""
    area = h * w
    dominant = _pick_dominant_color(targets)
    patch = np.full((h, w), dominant, dtype=int)
    all_coords = [(r, c) for r in range(h) for c in range(w)]

    max_target_cells = area // 2
    min_target_cells = NUM_TARGETS if require_all_targets else 0
    if max_target_cells < min_target_cells:
        raise ValueError("kept component too small for requested target coverage")

    if require_all_targets:
        lo = max(NUM_TARGETS, max_target_cells - max(1, area // 6))
        hi = max_target_cells
        target_count = random.randint(lo, hi)
    else:
        target_count = random.randint(min_target_cells, max_target_cells)

    target_coords = random.sample(all_coords, target_count) if target_count > 0 else []
    target_colors = _assign_target_colors(target_count, targets, require_all=require_all_targets) if target_count > 0 else []
    for (r, c), color in zip(target_coords, target_colors):
        patch[r, c] = color

    residual_coords = [p for p in all_coords if p not in set(target_coords)]
    alt_colors = _other_non_legend_colors(targets, dominant)
    max_mutate = min(KEEP_MUTATE_MAX, max(0, len(residual_coords) // 3))
    if alt_colors and max_mutate > 0:
        n_mutate = random.randint(1, max_mutate)
        for r, c in random.sample(residual_coords, n_mutate):
            patch[r, c] = random.choice(alt_colors)

    assert _target_pixel_count(np.argwhere(patch != BG), patch, set(targets)) * 2 <= area
    if require_all_targets:
        present = {int(patch[r, c]) for r, c in target_coords}
        assert set(targets).issubset(present)
    return patch


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input/output pair; return input or None on failure."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    matching_count = component_count - NUM_KEPT
    roles = [True] * matching_count + [False] * NUM_KEPT
    random.shuffle(roles)

    forced_partial_match_used = False
    forced_all_keep_used = False

    for idx, matching in enumerate(roles):
        h, w = _component_dims(matching)
        try:
            if matching:
                require_all_targets = forced_partial_match_used and random.random() < 0.35
                in_patch, out_patch = _component_patch_matching(
                    h, w, targets, require_all_targets=require_all_targets,
                )
                if not require_all_targets:
                    forced_partial_match_used = True
            else:
                remaining_keeps = sum(not r for r in roles[idx:])
                require_all_targets = (not forced_all_keep_used) and remaining_keeps == 1
                if (not forced_all_keep_used) and random.random() < 0.45:
                    require_all_targets = True
                patch = _component_patch_keep(h, w, targets, require_all_targets=require_all_targets)
                in_patch = patch
                out_patch = patch.copy()
                if require_all_targets:
                    forced_all_keep_used = True
        except ValueError:
            return None

        try:
            r, c = random_free_location(
                inp, np.zeros_like(in_patch), transparent=BG, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, in_patch, r, c)
        paste(out, out_patch, r, c)

    if not forced_partial_match_used or not forced_all_keep_used:
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
