# Variant of abc82100
# Dimension: D2 — 规则示例的编码格式 (symbol_instructions)
# Change: 规则示例由相邻的源色-目标色像素对加上附近的teal模板图案构成，颜色对直接编码映射关系... → 图例区域集中编码
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题里，每条规则是“散落在场景中的局部示例”: 相邻 source/target 色块旁边挂一个 teal 模板。
#      solver 逐像素扫描整张图，在任何位置都可能发现一条规则。
#    - 变体把所有规则集中到左上角图例条带。图例每条规则固定占 3 行:
#        * 第 2 行前两列分别是 source color、target color。
#        * 右侧 3x4 窗口中放一个 teal 小模板；该窗口的锚点固定为局部坐标 (1,1)。
#      所有规则上下紧贴堆叠，图例下方留一整行黑色分隔线；其余散落 source 像素只出现在右下工作区。
#    - 小例子追踪:
#        * 原逻辑: 在任意位置看到 [红][黄] 相邻，再沿方向找到附近 teal 连通块，就学到“红 -> 黄 + 该模板”。
#        * 新逻辑: 先截出左上图例，按 3 行一组读取 [红][黄] 和右侧 teal 模板，就学到同样的“红 -> 黄 + 该模板”。
#        * 差异: 规则发现不再依赖全图局部搜寻，而是先识别集中图例，再查表应用到散落源像素。
# b) 复杂度校准
#    - 仍保留 1-4 条规则、每条规则 3-5 个散落源像素、每条规则独立模板，避免退化成单规则单模板。
#    - 模板限制在 3x4 小窗内，但要求:
#        * 8 连通
#        * 包含固定锚点
#        * 触及顶行和底行
#      这样模板仍有明显形状差异，不会因为过小而看不出“查表 + 平移盖章”。
# c) 隐式假设审查（相对原 main 的变化）
#    - 原 `_discover_rules()` 假设规则示例散布在全图，且 anchor 由 source->target 的延长线决定；变体中这个前提失效。
#      新 solver 改为 `_decode_legend_rules()`，只从左上图例读取规则。
#    - 原逻辑把所有 source 颜色像素都视作“可能被替换的散落源像素”，但用 `legend_pixels` 排除局部示例里的 source/target。
#      变体里 legend 是矩形区域，因此排除条件改成“位于 legend rectangle 内”。
#    - 新 solver 新增假设:
#        * 图例从左上角开始，按 3 行为一条规则连续堆叠。
#        * 图例下方存在一整行背景色分隔。
#        * 图例宽度固定为 6 列，其中后 4 列是模板窗。
#      generator 会显式构造并验证这些条件。
# d) 放置合法性
#    - 图例始终贴左上角；图例下方保留 1 行空白；散落源像素和旁观像素只放在右下工作区。
#    - 散落源像素之间保持 4 邻隔离，防止被误看成新的局部规则示例。
#    - teal 只出现在图例里，避免回退成原题“在场景中找附近 teal 模板”。
#    - 旁观像素避开所有输出盖章足迹，确保其在输出中稳定保留。
# e) 视觉可观察性
#    - 人类会先看到左上角整齐的“颜色对 + 小模板”清单，再看到右下若干同色单像素，输出则把这些单像素替换成对应颜色模板。
#    - 规则的关键分支来自多条图例行，而不是依赖颜色 id 排序或不可见元数据。
# f) 可视化整洁度
#    - 输入被分成“左上图例”和“右下工作区”两块，中间有空白分隔，不会出现图例、teal、散点混在同一区域的杂乱感。
#    - 图例模板统一放在 3x4 小窗内，便于肉眼对照。
# g) 颜色变化可推断性
#    - 每条规则直接在图例里同时给出 source 与 target 颜色，并伴随模板结构，因此颜色变化始终有显式可见的配对线索。
#    - 颜色种类最多 8 个非黑非 teal 颜色，且一题中最多 4 条规则，查表负担可控。

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import get_bg, label_mask


PATTERN_COLOR = 8
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]

GRID_DIM_MAX = 20
MIN_HEIGHT_BY_RULES = {1: 9, 2: 12, 3: 15, 4: 18}
MIN_WIDTH_BY_RULES = {1: 11, 2: 13, 3: 16, 4: 18}

LEGEND_RULE_HEIGHT = 3
LEGEND_SEPARATOR = 1
TEMPLATE_H = 3
TEMPLATE_W = 4
TEMPLATE_COL0 = 2
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
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    masks = list(_valid_stencil_masks())
    rules: list[dict] = []
    for i in range(num_rules):
        src = colors[2 * i]
        tgt = colors[2 * i + 1]
        mask = random.choice(masks)
        rel = np.argwhere(mask).astype(int) - np.array(ANCHOR_LOCAL, dtype=int)
        rules.append(
            {
                "src": int(src),
                "tgt": int(tgt),
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
        legend[r0 + 1, 1] = int(rule["tgt"])
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


def _decode_legend_rules(grid: np.ndarray, bg: int, pattern_color: int = PATTERN_COLOR) -> tuple[list[dict], int]:
    legend_h = _detect_legend_height(grid, bg)
    if legend_h < LEGEND_RULE_HEIGHT or grid.shape[1] < LEGEND_WIDTH:
        raise ValueError("grid too small for legend strip")

    rules: list[dict] = []
    for r0 in range(0, legend_h, LEGEND_RULE_HEIGHT):
        src = int(grid[r0 + 1, 0])
        tgt = int(grid[r0 + 1, 1])
        if src in (int(bg), int(pattern_color)) or tgt in (int(bg), int(pattern_color)) or src == tgt:
            raise ValueError("invalid legend color pair")
        window = grid[r0 : r0 + TEMPLATE_H, TEMPLATE_COL0 : TEMPLATE_COL0 + TEMPLATE_W]
        mask = window == int(pattern_color)
        if not bool(mask[ANCHOR_LOCAL]):
            raise ValueError("legend stencil must include fixed anchor")
        rel = np.argwhere(mask).astype(int) - np.array(ANCHOR_LOCAL, dtype=int)
        rules.append({"source_color": src, "target_color": tgt, "rel": np.asarray(rel, dtype=int)})
    return rules, legend_h


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a top-left legend strip to look up which teal stencil each source color expands into.

    Input structure:
    - Black background with a left-top legend made of stacked 3-row entries.
    - In each entry, the middle-row first two cells give source and target colors,
      and the 3x4 teal patch on the right is that rule's miniature stencil.

    Rule:
    - Decode every legend entry into a source_color -> (target_color, stencil) rule.
    - For each scattered source pixel outside the legend, translate the rule's teal
      stencil so its fixed anchor lands on that source pixel, then recolor it to target_color.
    - Erase the entire legend and all scattered source pixels after stamping; keep
      unrelated bystander pixels unchanged.

    Key insight: The mapping is no longer hidden in local scene fragments; it is a
    centralized lookup table, so the solver must isolate the legend first and then
    apply the listed stencil/color pair to matching source pixels elsewhere.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules, legend_h = _decode_legend_rules(g, bg=bg, pattern_color=PATTERN_COLOR)
    source_colors = {int(rule["source_color"]) for rule in rules}

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != int(PATTERN_COLOR)))):
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
            positions.append((r, c))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
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
    used_colors = {int(rule["src"]) for rule in rules} | {int(rule["tgt"]) for rule in rules}
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
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
            color = random.choice(bystander_colors)
            placed.append((r, c, int(color)))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
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
