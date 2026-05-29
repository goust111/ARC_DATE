# Combo variant of abc82100 (index 49)
# combo_spec: D2/0+D16/1
# Name: 三像素编码+图例区域集中
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题在整张图里散放局部示例：某个 source 色像素旁边紧挨着 target 色像素，
#      再沿同方向找到靠近 teal 模板的 anchor；solver 要全图搜索这种局部证据。
#    - 融合后，所有规则被集中到左上角图例区。每条规则固定占 3x7：
#        * 中间行前三列是 `source, gray(5), target` 三像素编码；
#        * 右侧 3x4 小窗里放 teal 模板，锚点固定为窗口局部坐标 (1,1)。
#      图例下方保留 1 行黑色分隔，图例右侧也保留 1 列黑色分隔；散落 source 像素和旁观像素只放在右下工作区。
#    - 小例子：
#        * 原逻辑：看到 `红 黄` 相邻且旁边连着某个 teal 形状，于是学到“红 -> 用黄重着色该形状”。
#        * 新逻辑：先在左上图例读到 `红 灰 黄`，再从同条图例右侧的小 teal 形状读出模板；
#          之后在工作区里遇到任意红色单像素，就把该模板平移到它身上并改成黄色。
#        * 差异：规则发现从“场景局部检索”变为“先定位集中图例，再查表应用”，同时颜色对由二像素改成三像素编码。
# b) 复杂度校准
#    - 保留 1-4 条规则，每条规则 3-5 个散落源像素，确保同一 source 颜色在工作区有重复支撑。
#    - 模板限制在统一的 3x4 窗口内，但要求 8 连通、包含固定锚点、同时触及顶行和底行，
#      这样模板既足够小便于图例排布，又不会退化成过于单调的点状标记。
#    - 分隔符颜色固定为灰色(5)，不参与其他对象，视觉上能稳定提示“三像素编码”的边界。
# c) 隐式假设审查
#    - 原 main 依赖 `_discover_rules()` 在全图寻找 source-target 相邻对；融合后这个前提失效，
#      必须改成 `_decode_legend_rules()` 只读取左上图例。
#    - 原 main 用 `legend_pixels` 排除局部示例；融合后图例是整块矩形，因此排除条件改成
#      “像素位于 legend rectangle 内”。
#    - 原 main 把所有非背景非 teal 颜色都当作潜在普通像素；融合后灰色(5)是结构分隔符，
#      必须在保留逻辑里排除。
#    - 新 solver 额外假设：
#        * 图例从左上角开始，连续堆叠，行高是 3 的倍数；
#        * 图例后面紧跟一行空白分隔，因此可以从顶端扫描到 legend 高度；
#        * 每条规则的模板窗固定在列 3-6，且包含固定锚点。
#      生成器会显式构造这些条件。
# d) 放置合法性
#    - 图例固定贴左上角，工作区从 `legend_h + 1` 行、`legend_w + 1` 列开始，避免图例与散点混杂。
#    - 散落 source 像素之间保持 4 邻隔离，防止视觉上粘连成额外结构。
#    - 旁观像素避开所有未来输出盖章足迹，因此在输出中一定保持原样。
#    - teal 只出现在图例窗口内，避免退回原题那种“场景中另有 teal 模板可搜索”的语义。
# e) 视觉可观察性
#    - 人类会先看到左上角整齐堆叠的多条图例，每条都有明显的 `彩色-灰色-彩色` 编码和一个 teal 小模板；
#      再看到右下工作区散落的单像素 source。输出中这些 source 消失，换成对应 target 颜色的模板印章。
#    - 多条规则共享同一种编码格式、重复使用相同 source 颜色的多个散点，因此可以从少量样例中归纳出“查图例后盖章”的机制。
# f) 可视化整洁度
#    - 图例与工作区由一整行、一整列黑色分隔带切开，避免同一区域混入模板、结构码和散点。
#    - 模板都被裁进一致的 3x4 窗口，图例视觉规整；工作区只保留散点与少量旁观像素，不会过于拥挤。
# g) 变化的可推断性
#    - 颜色变化并非无提示发生：每条规则的 source 与 target 颜色在图例中同排明示，中间灰色像素表明这是一个三段编码。
#    - 非特殊颜色总量受控，source 颜色唯一，target 颜色尽量不重复且优先不与任何 source 冲突，
#      这样查表映射对人类和 solver 都保持清晰。
# h) 新 solver 假设与生成器对齐
#    - `_detect_legend_height()` 依赖“图例后第一行为空白”：由 `_build_input()` 和工作区放置范围保证。
#    - `_decode_legend_rules()` 依赖每条图例中行的灰色分隔符和固定模板窗：由 `_legend_patch()` 保证。
#    - `main()` 依赖工作区不存在额外灰色或 teal：由 `_build_rules()` 的颜色集和 `_build_input()` 的对象类型保证。

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import get_bg, label_mask


PATTERN_COLOR = 8
SEPARATOR_COLOR = 5
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 9]

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]

GRID_DIM_MAX = 20
MIN_HEIGHT_BY_RULES = {1: 9, 2: 12, 3: 15, 4: 18}
MIN_WIDTH_BY_RULES = {1: 12, 2: 13, 3: 16, 4: 18}

LEGEND_RULE_HEIGHT = 3
LEGEND_SEPARATOR = 1
TEMPLATE_H = 3
TEMPLATE_W = 4
TEMPLATE_COL0 = 3
ANCHOR_LOCAL = (1, 1)
LEGEND_WIDTH = TEMPLATE_COL0 + TEMPLATE_W

STENCIL_AREA_RANGE = (3, 6)
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 5
BYSTANDER_MAX = 5

SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


@lru_cache(maxsize=None)
def _valid_stencil_masks() -> tuple[np.ndarray, ...]:
    """All connected 3x4 masks that contain the fixed anchor and touch top/bottom rows."""
    masks: list[np.ndarray] = []
    anchor_index = ANCHOR_LOCAL[0] * TEMPLATE_W + ANCHOR_LOCAL[1]
    for area in range(STENCIL_AREA_RANGE[0], STENCIL_AREA_RANGE[1] + 1):
        for cells in combinations(range(TEMPLATE_H * TEMPLATE_W), area):
            if anchor_index not in cells:
                continue
            mask = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=bool)
            for cell in cells:
                mask[cell // TEMPLATE_W, cell % TEMPLATE_W] = True
            if not mask[0, :].any() or not mask[-1, :].any():
                continue
            _labels, count = label_mask(mask, connectivity=8)
            if int(count) == 1:
                masks.append(mask)
    return tuple(masks)


def _build_rules(num_rules: int) -> list[dict]:
    colors = NON_SPECIAL_COLORS[:]
    random.shuffle(colors)
    source_colors = colors[:num_rules]
    masks = list(_valid_stencil_masks())

    rules: list[dict] = []
    used_targets: set[int] = set()
    for src in source_colors:
        candidates = [c for c in NON_SPECIAL_COLORS if c != src]
        preferred = [c for c in candidates if c not in source_colors and c not in used_targets]
        secondary = [c for c in candidates if c not in source_colors]
        target_pool = preferred or secondary or [c for c in candidates if c not in used_targets] or candidates
        tgt = int(random.choice(target_pool))
        used_targets.add(tgt)
        mask = random.choice(masks)
        rel = np.argwhere(mask).astype(int) - np.array(ANCHOR_LOCAL, dtype=int)
        rules.append(
            {
                "src": int(src),
                "tgt": tgt,
                "mask": mask.copy(),
                "rel": np.asarray(rel, dtype=int),
            }
        )
    return rules


def _legend_height(num_rules: int) -> int:
    return LEGEND_RULE_HEIGHT * int(num_rules)


def _work_area(num_rules: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    top = _legend_height(num_rules) + LEGEND_SEPARATOR
    left = LEGEND_WIDTH + LEGEND_SEPARATOR
    return top, h, left, w


def _legend_patch(rules: list[dict]) -> np.ndarray:
    legend = np.zeros((_legend_height(len(rules)), LEGEND_WIDTH), dtype=int)
    for i, rule in enumerate(rules):
        r0 = i * LEGEND_RULE_HEIGHT
        legend[r0 + 1, 0] = int(rule["src"])
        legend[r0 + 1, 1] = SEPARATOR_COLOR
        legend[r0 + 1, 2] = int(rule["tgt"])
        window = legend[r0 : r0 + TEMPLATE_H, TEMPLATE_COL0 : TEMPLATE_COL0 + TEMPLATE_W]
        window[rule["mask"]] = PATTERN_COLOR
    return legend


def _detect_legend_height(grid: np.ndarray, bg: int) -> int:
    last_non_bg = -1
    seen = False
    for r in range(grid.shape[0]):
        if np.any(grid[int(r), :] != int(bg)):
            seen = True
            last_non_bg = int(r)
        elif seen:
            break
    if last_non_bg < 0:
        raise ValueError("legend not found")
    legend_h = last_non_bg + 1
    if legend_h % LEGEND_RULE_HEIGHT != 0:
        raise ValueError("legend height is not a multiple of 3")
    return legend_h


def _decode_legend_rules(
    grid: np.ndarray,
    bg: int,
    pattern_color: int = PATTERN_COLOR,
    separator_color: int = SEPARATOR_COLOR,
) -> tuple[list[dict], int]:
    legend_h = _detect_legend_height(grid, bg)
    if legend_h < LEGEND_RULE_HEIGHT or grid.shape[1] < LEGEND_WIDTH:
        raise ValueError("grid too small for legend strip")

    rules: list[dict] = []
    for r0 in range(0, legend_h, LEGEND_RULE_HEIGHT):
        src = int(grid[r0 + 1, 0])
        sep = int(grid[r0 + 1, 1])
        tgt = int(grid[r0 + 1, 2])
        if sep != int(separator_color):
            raise ValueError("legend separator missing")
        if src in (int(bg), int(pattern_color), int(separator_color)):
            raise ValueError("invalid legend source color")
        if tgt in (int(bg), int(pattern_color), int(separator_color)) or tgt == src:
            raise ValueError("invalid legend target color")

        window = grid[r0 : r0 + TEMPLATE_H, TEMPLATE_COL0 : TEMPLATE_COL0 + TEMPLATE_W]
        mask = window == int(pattern_color)
        if not bool(mask[ANCHOR_LOCAL]):
            raise ValueError("legend stencil must include fixed anchor")
        rel = np.argwhere(mask).astype(int) - np.array(ANCHOR_LOCAL, dtype=int)
        rules.append(
            {
                "source_color": src,
                "target_color": tgt,
                "rel": np.asarray(rel, dtype=int),
            }
        )
    return rules, legend_h


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a centralized source-separator-target legend to expand matching source pixels into recolored teal stencils.

    Input structure:
    - Black background with a top-left legend made of stacked 3-row entries and a separated work area below-right.
    - In each entry, the middle row begins with three pixels `source, gray(5), target`,
      and the 3x4 window to the right contains the teal (8) stencil for that rule.

    Rule:
    - Decode each legend entry into a mapping from source color to target color plus a fixed stencil shape.
    - For every scattered source pixel outside the legend, translate that rule's stencil so the legend's fixed anchor
      lands on the source pixel, then recolor the stamped cells to the target color.
    - Remove the full legend, the gray separators, the teal template pixels, and the consumed scattered source pixels;
      preserve unrelated bystander pixels.

    Key insight: The informative cue is now a compact lookup table in the corner: the gray pixel marks that the two
    colored pixels are one encoded rule, and the adjacent teal mini-template shows the exact shape to stamp elsewhere.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules, legend_h = _decode_legend_rules(
        g,
        bg=bg,
        pattern_color=PATTERN_COLOR,
        separator_color=SEPARATOR_COLOR,
    )
    source_colors = {int(rule["source_color"]) for rule in rules}

    out = np.full_like(g, int(bg))

    keep_mask = (g != int(bg)) & (g != int(PATTERN_COLOR)) & (g != int(SEPARATOR_COLOR))
    for r, c in map(tuple, np.argwhere(keep_mask)):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if r < legend_h and c < LEGEND_WIDTH:
            continue
        if color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        rel = np.asarray(rule["rel"], dtype=int)
        for sr, sc in map(tuple, np.argwhere(g == int(rule["source_color"]))):
            sr = int(sr)
            sc = int(sc)
            if sr < legend_h and sc < LEGEND_WIDTH:
                continue
            rows = sr + rel[:, 0]
            cols = sc + rel[:, 1]
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


def _stamp_cells(sr: int, sc: int, rel: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    cells: set[tuple[int, int]] = set()
    for rr, cc in np.asarray(rel, dtype=int):
        nr = int(sr) + int(rr)
        nc = int(sc) + int(cc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    return cells


def _place_scattered(rules: list[dict], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    num_rules = len(rules)
    work_r0, _work_r1, work_c0, _work_c1 = _work_area(num_rules, shape)
    work_h = h - work_r0
    work_w = w - work_c0
    if min(work_h, work_w) <= 0:
        return False

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    sc_max = min(
        SCATTERED_MAX_CAP,
        max(SCATTERED_MIN, (work_h * work_w) // max(6, 4 * num_rules)),
    )

    for rule in rules:
        rel = np.asarray(rule["rel"], dtype=int)
        min_rr = int(rel[:, 0].min())
        min_cc = int(rel[:, 1].min())
        row_lo = work_r0 - min_rr
        col_lo = work_c0 - min_cc
        if row_lo >= h or col_lo >= w:
            return False

        n_sc = random.randint(SCATTERED_MIN, sc_max)
        positions: list[tuple[int, int]] = []
        for _ in range(n_sc * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= n_sc:
                break
            r = random.randint(row_lo, h - 1)
            c = random.randint(col_lo, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions

    return True


def _stamp_footprint(rules: list[dict], shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for rule in rules:
        for sr, sc in rule["scattered"]:
            cells |= _stamp_cells(sr, sc, rule["rel"], shape)
    return cells


def _place_bystanders(rules: list[dict], shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    h, w = map(int, shape)
    num_rules = len(rules)
    work_r0, _work_r1, work_c0, _work_c1 = _work_area(num_rules, shape)
    used_colors = {int(rule["src"]) for rule in rules} | {int(rule["tgt"]) for rule in rules} | {SEPARATOR_COLOR}
    bystander_colors = [c for c in NON_SPECIAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    occupied = {tuple(map(int, pos)) for rule in rules for pos in rule["scattered"]}
    reserved: set[tuple[int, int]] = set()
    _reserve_4neighbors(reserved, list(occupied))
    forbidden = _stamp_footprint(rules, shape)

    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed: list[tuple[int, int, int]] = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(work_r0, h - 1)
            c = random.randint(work_c0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in forbidden:
                continue
            color = int(random.choice(bystander_colors))
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules: list[dict], bystanders: list[tuple[int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for nr, nc in _stamp_cells(sr, sc, rule["rel"], shape):
                out[int(nr), int(nc)] = int(rule["tgt"])
    return out


def _build_input(rules: list[dict], bystanders: list[tuple[int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    legend = _legend_patch(rules)
    inp[: legend.shape[0], : legend.shape[1]] = legend
    for rule in rules:
        for r, c in rule["scattered"]:
            inp[int(r), int(c)] = int(rule["src"])
    for r, c, color in bystanders:
        inp[int(r), int(c)] = int(color)
    return inp


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    h = random.randint(MIN_HEIGHT_BY_RULES[num_rules], GRID_DIM_MAX)
    w = random.randint(MIN_WIDTH_BY_RULES[num_rules], GRID_DIM_MAX)
    shape = (h, w)

    rules = _build_rules(num_rules)
    if not _place_scattered(rules, shape):
        return None
    bystanders = _place_bystanders(rules, shape)

    inp = _build_input(rules, bystanders, shape)
    expected = _build_output(rules, bystanders, shape)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
